#include "geozl/gpu.h"

#include "geozl/ctids.h"
#include "geozl/geozl.h"

#include "modules.h"
#include "pfor_decode.h"
#include "pivco_decode.h"
#include "runtime_internal.h"

#include "openzl/decompress/decode_frameheader.h"
#include "openzl/zl_decompress.h"

#include <limits.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Frames route by their last codec. A frame the device path cannot capture
// falls back to the host, so routing only has to be right in the common case.
enum { ROUTE_HOST, ROUTE_PFOR, ROUTE_PIVCO, ROUTE_FALLBACK };

enum { SOURCE_DEVICE, SOURCE_HOST, SOURCE_FALLBACK, SOURCES };

#define PLACE_WIDTHS 5
#define MAX_PENDING 8
// CUDA caps grid Y at 65535 and place covers four rows per group. Its threads
// step across a row 128 samples at a time, which must not wrap 32 bits.
#define PLACE_MAX_ROWS ((size_t)65535 * 4)
#define PLACE_MAX_WIDTH (UINT32_MAX - 128u)
// Device frames share 32-bit sample and byte offsets in their kernels.
#define DEVICE_BUDGET ((size_t)UINT32_MAX - 4096)

typedef struct {
  ZL_DCtx *dctx;
  PforCapture pfor;
  PivcoCapture pivco;
  uint8_t *scratch;
  size_t scratchBytes;
  int ready;
} Worker;

struct GeozlGpu {
  GeozlGpuBackendId backend;
  int device;
  size_t threads;
  size_t maxDecodedBytes;
  int verify;
  GeozlGpuProgram *place[PLACE_WIDTHS];
  GeozlPforGpuDecoder *pfor[2];
  GeozlPivcoGpuDecoder *pivco[2];
  GeozlGpuArena staging;
  GeozlGpuMemory *scratch;
  Worker *workers;
  GeozlGpuBatch *pending[MAX_PENDING];
  size_t pendingCount;
  GeozlGpuMemory *target;
};

// Mirrors place_copy in gpu/kernels/programs/place.slang. Counts are samples.
typedef struct {
  uint32_t source;
  uint32_t sourceRow;
  uint32_t sourceStep;
  uint32_t destination;
  uint32_t destinationRow;
  uint32_t destinationStep;
  uint32_t width;
  uint32_t rows;
} PlaceCopy;

static const char *const placeKernels[] = {"place"};

static int backend_of(int deviceType, GeozlGpuBackendId *backend) {
  switch (deviceType) {
  case GEOZL_GPU_DEVICE_CUDA:
    *backend = GEOZL_GPU_BACKEND_CUDA;
    break;
  case GEOZL_GPU_DEVICE_METAL:
    *backend = GEOZL_GPU_BACKEND_METAL;
    break;
  default:
    return 0;
  }
  return geozl_gpu_backend_available(*backend);
}

static const GeozlGpuModule *find_module(GeozlGpuBackendId backend,
                                         const char *program, size_t width) {
  (void)backend;
  char name[32];
  snprintf(name, sizeof(name), "%s%zu", program, width);
  for (size_t i = 0; i < geozl_gpu_module_count; ++i) {
    if (strcmp(geozl_gpu_modules[i].name, name) == 0)
      return &geozl_gpu_modules[i];
  }
  return NULL;
}

static int place_slot(size_t sampleBytes) {
  switch (sampleBytes) {
  case 1:
    return 0;
  case 2:
    return 1;
  case 4:
    return 2;
  case 8:
    return 3;
  case 16:
    return 4;
  default:
    return -1;
  }
}

static GeozlGpuProgram *place_program(GeozlGpu *gpu, size_t sampleBytes,
                                      char *error, size_t errorCapacity) {
  const int slot = place_slot(sampleBytes);
  if (gpu->place[slot] != NULL)
    return gpu->place[slot];
  const GeozlGpuModule *module =
      find_module(gpu->backend, "place", sampleBytes);
  if (module == NULL) {
    geozl_gpu_error(error, errorCapacity, "GPU kernels are missing");
    return NULL;
  }
  gpu->place[slot] = geozl_gpu_program_create(
      gpu->backend, module->data, module->size, gpu->device, placeKernels, 1,
      error, errorCapacity);
  return gpu->place[slot];
}

static GeozlPforGpuDecoder *pfor_decoder(GeozlGpu *gpu, size_t width,
                                         char *error, size_t errorCapacity) {
  if (gpu->pfor[width - 1] == NULL) {
    const GeozlGpuModule *module = find_module(gpu->backend, "pfor", width);
    if (module == NULL) {
      geozl_gpu_error(error, errorCapacity, "GPU kernels are missing");
      return NULL;
    }
    gpu->pfor[width - 1] =
        geozl_pfor_gpu_decoder_create(gpu->backend, module->data, module->size,
                                      width, gpu->device, error, errorCapacity);
  }
  return gpu->pfor[width - 1];
}

static GeozlPivcoGpuDecoder *pivco_decoder(GeozlGpu *gpu, size_t width,
                                           char *error, size_t errorCapacity) {
  if (gpu->pivco[width - 1] == NULL) {
    const GeozlGpuModule *module = find_module(gpu->backend, "pivco", width);
    if (module == NULL) {
      geozl_gpu_error(error, errorCapacity, "GPU kernels are missing");
      return NULL;
    }
    gpu->pivco[width - 1] = geozl_pivco_gpu_decoder_create(
        gpu->backend, module->data, module->size, width, gpu->device, error,
        errorCapacity);
  }
  return gpu->pivco[width - 1];
}

static void worker_destroy(Worker *worker) {
  ZL_DCtx_free(worker->dctx);
  pfor_capture_destroy(&worker->pfor);
  pivco_capture_destroy(&worker->pivco);
  free(worker->scratch);
  *worker = (Worker){0};
}

// Without verification the compressed checksum is skipped as on the host.
static int worker_init(Worker *worker, int verify) {
  if (worker->ready)
    return 0;
  const ZL_TernaryParam check =
      verify ? ZL_TernaryParam_enable : ZL_TernaryParam_disable;
  worker->dctx = ZL_DCtx_create();
  if (worker->dctx == NULL || pfor_capture_init(&worker->pfor, NULL, 0) != 0 ||
      pivco_capture_init(&worker->pivco, NULL, 0) != 0 ||
      ZL_isError(geozl_register_decoders(worker->dctx)) ||
      ZL_isError(
          ZL_DCtx_setParameter(worker->dctx, ZL_DParam_stickyParameters, 1)) ||
      ZL_isError(ZL_DCtx_setParameter(
          worker->dctx, ZL_DParam_checkCompressedChecksum, check)) ||
      ZL_isError(ZL_DCtx_setParameter(worker->dctx,
                                      ZL_DParam_checkContentChecksum, check)) ||
      ZL_isError(ZL_DCtx_setParameter(
          worker->pfor.dctx, ZL_DParam_checkCompressedChecksum, check)) ||
      ZL_isError(ZL_DCtx_setParameter(
          worker->pivco.dctx, ZL_DParam_checkCompressedChecksum, check))) {
    worker_destroy(worker);
    return 1;
  }
  worker->ready = 1;
  return 0;
}

int geozl_gpu_available(int deviceType) {
  GeozlGpuBackendId backend;
  return backend_of(deviceType, &backend);
}

GeozlGpu *geozl_gpu_create(int deviceType, int device, size_t threads,
                           size_t maxDecodedBytes, int verifyChecksums,
                           char *error, size_t errorCapacity) {
  if (error != NULL && errorCapacity != 0)
    error[0] = '\0';
  GeozlGpuBackendId backend;
  if (!backend_of(deviceType, &backend)) {
    geozl_gpu_error(error, errorCapacity, "GPU backend is not built in");
    return NULL;
  }
  if (device < 0) {
    geozl_gpu_error(error, errorCapacity, "invalid GPU device");
    return NULL;
  }
  if (maxDecodedBytes == 0) {
    geozl_gpu_error(error, errorCapacity, "invalid decoded byte limit");
    return NULL;
  }
  GeozlGpu *gpu = calloc(1, sizeof(*gpu));
  if (gpu == NULL) {
    geozl_gpu_error(error, errorCapacity, "out of memory");
    return NULL;
  }
  gpu->backend = backend;
  gpu->device = device;
  gpu->threads = threads == 0 ? 1 : threads;
  gpu->maxDecodedBytes = maxDecodedBytes;
  gpu->verify = verifyChecksums != 0;
  gpu->workers = calloc(gpu->threads, sizeof(*gpu->workers));
  // The one-byte place program opens the device now, so a missing driver
  // fails here, and it owns the staging.
  GeozlGpuProgram *program =
      gpu->workers == NULL ? NULL : place_program(gpu, 1, error, errorCapacity);
  if (program == NULL) {
    if (gpu->workers == NULL)
      geozl_gpu_error(error, errorCapacity, "out of memory");
    geozl_gpu_destroy(gpu);
    return NULL;
  }
  geozl_gpu_arena_init(&gpu->staging, program);
  return gpu;
}

static int finish_pending(GeozlGpu *gpu, char *error, size_t errorCapacity) {
  int failed = 0;
  for (size_t i = 0; i < gpu->pendingCount; ++i) {
    if (!failed &&
        geozl_gpu_batch_wait(gpu->pending[i], error, errorCapacity) != 0)
      failed = 1;
    geozl_gpu_batch_destroy(gpu->pending[i]);
  }
  gpu->pendingCount = 0;
  geozl_gpu_memory_destroy(gpu->target);
  gpu->target = NULL;
  return failed;
}

void geozl_gpu_destroy(GeozlGpu *gpu) {
  if (gpu == NULL)
    return;
  finish_pending(gpu, NULL, 0);
  geozl_gpu_memory_destroy(gpu->scratch);
  if (gpu->place[0] != NULL)
    geozl_gpu_arena_destroy(&gpu->staging);
  if (gpu->workers != NULL) {
    for (size_t i = 0; i < gpu->threads; ++i)
      worker_destroy(&gpu->workers[i]);
  }
  free(gpu->workers);
  for (size_t i = 0; i < 2; ++i) {
    geozl_pfor_gpu_decoder_destroy(gpu->pfor[i]);
    geozl_pivco_gpu_decoder_destroy(gpu->pivco[i]);
  }
  for (size_t i = 0; i < PLACE_WIDTHS; ++i)
    geozl_gpu_program_destroy(gpu->place[i]);
  free(gpu);
}

int geozl_gpu_synchronize(GeozlGpu *gpu, char *error, size_t errorCapacity) {
  if (error != NULL && errorCapacity != 0)
    error[0] = '\0';
  if (gpu == NULL)
    return geozl_gpu_error(error, errorCapacity, "invalid GPU decoder");
  return finish_pending(gpu, error, errorCapacity);
}

GeozlGpuMemory *geozl_gpu_alloc(GeozlGpu *gpu, size_t bytes, char *error,
                                size_t errorCapacity) {
  if (error != NULL && errorCapacity != 0)
    error[0] = '\0';
  if (gpu == NULL) {
    geozl_gpu_error(error, errorCapacity, "invalid GPU decoder");
    return NULL;
  }
  return geozl_gpu_memory_create(gpu->place[0], bytes, error, errorCapacity);
}

void *geozl_gpu_pointer(const GeozlGpuMemory *memory) {
  return geozl_gpu_memory_data(memory);
}

void geozl_gpu_free(GeozlGpuMemory *memory) {
  geozl_gpu_memory_destroy(memory);
}

static int route_of(const GeozlGpuFrame *frame) {
  DFH_Struct header = {0};
  DFH_init(&header);
  int route = ROUTE_HOST;
  const ZL_Report frameHeader =
      DFH_decodeFrameHeader(&header, frame->data, frame->size);
  if (!ZL_isError(frameHeader) &&
      (size_t)ZL_validResult(frameHeader) <= frame->size) {
    const size_t used = (size_t)ZL_validResult(frameHeader);
    const ZL_Report chunk = DFH_decodeChunkHeader(
        &header, (const uint8_t *)frame->data + used, frame->size - used);
    const size_t count = VECTOR_SIZE(header.nodes);
    if (!ZL_isError(chunk) && count != 0) {
      // OpenZL lists the last codec applied on decode at the end.
      const PublicTransformInfo last = VECTOR_AT(header.nodes, count - 1).trpid;
      if (last.trt == trt_custom &&
          (last.trid == GEOZL_CTID_PFOR ||
           last.trid == GEOZL_CTID_PLANAR_ZIGZAG_PFOR))
        route = ROUTE_PFOR;
      else if (last.trt == trt_custom &&
               (last.trid == GEOZL_CTID_PLANAR_ZIGZAG_TRANSPOSE_PIVCO ||
                last.trid == GEOZL_CTID_TRANSPOSE_PIVCO))
        route = ROUTE_PIVCO;
    }
  }
  DFH_destroy(&header);
  return route;
}

static size_t round_up(size_t value, size_t to) {
  return value > SIZE_MAX - (to - 1) ? SIZE_MAX : (value + to - 1) / to * to;
}

static int add_size(size_t *total, size_t value) {
  if (value > SIZE_MAX - *total)
    return 1;
  *total += value;
  return 0;
}

// Last byte past a strided rectangle starting at offset, or SIZE_MAX.
static size_t span_end(size_t offset, size_t rows, size_t rowBytes,
                       size_t width, size_t stepBytes, size_t sampleBytes) {
  size_t end = offset;
  if ((rows > 1 && rowBytes > (SIZE_MAX - end) / (rows - 1)))
    return SIZE_MAX;
  end += rows > 1 ? (rows - 1) * rowBytes : 0;
  if (width > 1 && stepBytes > (SIZE_MAX - end) / (width - 1))
    return SIZE_MAX;
  end += width > 1 ? (width - 1) * stepBytes : 0;
  return end > SIZE_MAX - sampleBytes ? SIZE_MAX : end + sampleBytes;
}

static int copies_valid(const GeozlGpuFrame frames[], size_t frameCount,
                        const GeozlGpuCopy copies[], size_t copyCount,
                        size_t sampleBytes, size_t targetBytes) {
  for (size_t i = 0; i < copyCount; ++i) {
    const GeozlGpuCopy *copy = &copies[i];
    if (copy->frame >= frameCount || copy->width == 0 || copy->rows == 0 ||
        copy->width > PLACE_MAX_WIDTH ||
        (copy->rows > 1 && copy->destinationRow == 0) ||
        (copy->width > 1 && copy->destinationStep == 0) ||
        copy->source % sampleBytes != 0 || copy->sourceRow % sampleBytes != 0 ||
        copy->sourceStep % sampleBytes != 0 ||
        copy->destination % sampleBytes != 0 ||
        copy->destinationRow % sampleBytes != 0 ||
        copy->destinationStep % sampleBytes != 0 ||
        span_end(copy->source, copy->rows, copy->sourceRow, copy->width,
                 copy->sourceStep,
                 sampleBytes) > frames[copy->frame].decodedBytes ||
        span_end(copy->destination, copy->rows, copy->destinationRow,
                 copy->width, copy->destinationStep, sampleBytes) > targetBytes)
      return 0;
  }
  return 1;
}

typedef struct {
  GeozlGpu *gpu;
  const GeozlGpuFrame *frames;
  size_t frameCount;
  uint8_t *routes;
  const size_t *slots;
  uint8_t *pforPayload;
  uint8_t *pivcoPayload;
  uint8_t *host;
  CapturedPforFrame *pforCaptured;
  CapturedFrame *pivcoCaptured;
  size_t elementBytes;
  size_t sampleBytes;
  atomic_size_t next;
  atomic_int failed;
  pthread_mutex_t lock;
  size_t errorFrame;
  char error[256];
} Pass;

typedef struct {
  Pass *pass;
  Worker *worker;
  int threaded;
} PassThread;

static int width_ok(size_t width, size_t elementBytes, size_t sampleBytes) {
  return width == elementBytes || width == sampleBytes;
}

static int host_decode(Worker *worker, const GeozlGpuFrame *frame, void *dst,
                       size_t elementBytes, size_t sampleBytes, char *error,
                       size_t errorCapacity) {
  ZL_OutputInfo info;
  const ZL_Report report = ZL_DCtx_decompressTyped(
      worker->dctx, &info, dst, frame->decodedBytes, frame->data, frame->size);
  if (ZL_isError(report))
    return geozl_gpu_error(error, errorCapacity, "OpenZL decode failed: %s",
                           ZL_DCtx_getErrorContextString(worker->dctx, report));
  if (info.type != ZL_Type_numeric ||
      !width_ok(info.fixedWidth, elementBytes, sampleBytes) ||
      info.decompressedByteSize != frame->decodedBytes)
    return geozl_gpu_error(error, errorCapacity,
                           "frame decodes to unexpected output");
  return 0;
}

static void fail_frame(Pass *pass, size_t frame, const char *message) {
  pthread_mutex_lock(&pass->lock);
  if (!atomic_load(&pass->failed) || frame < pass->errorFrame) {
    pass->errorFrame = frame;
    snprintf(pass->error, sizeof(pass->error), "%s", message);
  }
  atomic_store(&pass->failed, 1);
  pthread_mutex_unlock(&pass->lock);
}

// Valid frames that need more host transforms fall back. A capture error is a
// malformed frame and must not be handed to a different decoder path.
static void capture_frame(Pass *pass, Worker *worker, size_t i) {
  const GeozlGpuFrame *frame = &pass->frames[i];
  const size_t slotBytes = round_up(frame->size, 16);
  if (worker->scratchBytes < frame->decodedBytes) {
    free(worker->scratch);
    worker->scratchBytes = 0;
    worker->scratch = malloc(frame->decodedBytes);
    if (worker->scratch == NULL) {
      fail_frame(pass, i, "out of memory");
      return;
    }
    worker->scratchBytes = frame->decodedBytes;
  }
  if (pass->routes[i] == ROUTE_PFOR) {
    worker->pfor.payload = pass->pforPayload + pass->slots[i];
    worker->pfor.payloadSize = 0;
    worker->pfor.payloadCapacity = slotBytes;
    const PforCaptureResult result = pfor_capture_frame(
        &worker->pfor, &pass->pforCaptured[i], worker->scratch,
        frame->decodedBytes, frame->data, frame->size);
    if (result == PFOR_CAPTURE_FALLBACK || result == PFOR_CAPTURE_RAW)
      pass->routes[i] = ROUTE_FALLBACK;
    else if (result != PFOR_CAPTURE_CODEC)
      fail_frame(pass, i, "invalid compressed frame");
  } else {
    worker->pivco.payload = pass->pivcoPayload + pass->slots[i];
    worker->pivco.payloadSize = 0;
    worker->pivco.payloadCapacity = slotBytes;
    const PivcoCaptureResult result = pivco_capture_frame(
        &worker->pivco, &pass->pivcoCaptured[i], worker->scratch,
        frame->decodedBytes, frame->data, frame->size);
    if (result == PIVCO_CAPTURE_FALLBACK || result == PIVCO_CAPTURE_RAW)
      pass->routes[i] = ROUTE_FALLBACK;
    else if (result != PIVCO_CAPTURE_CODEC)
      fail_frame(pass, i, "invalid compressed frame");
  }
}

static void *run_pass(void *opaque) {
  PassThread *thread = opaque;
  Pass *pass = thread->pass;
  char message[224];
  for (;;) {
    const size_t i = atomic_fetch_add(&pass->next, 1);
    if (i >= pass->frameCount || atomic_load(&pass->failed))
      break;
    if (pass->routes[i] == ROUTE_HOST) {
      if (host_decode(thread->worker, &pass->frames[i],
                      pass->host + pass->slots[i], pass->elementBytes,
                      pass->sampleBytes, message, sizeof(message)) != 0)
        fail_frame(pass, i, message);
    } else {
      capture_frame(pass, thread->worker, i);
    }
  }
  return NULL;
}

// Workers take frames in order from a shared counter, so slow host frames do
// not leave threads idle. A worker without a thread runs on this one.
static int run_frames(GeozlGpu *gpu, Pass *pass, char *error,
                      size_t errorCapacity) {
  size_t threads =
      gpu->threads < pass->frameCount ? gpu->threads : pass->frameCount;
  PassThread *workers = calloc(threads, sizeof(*workers));
  pthread_t *handles = calloc(threads, sizeof(*handles));
  if (workers == NULL || handles == NULL) {
    free(workers);
    free(handles);
    return geozl_gpu_error(error, errorCapacity, "out of memory");
  }
  for (size_t i = 0; i < threads; ++i) {
    if (worker_init(&gpu->workers[i], gpu->verify) != 0) {
      threads = i;
      break;
    }
  }
  if (threads == 0) {
    free(workers);
    free(handles);
    return geozl_gpu_error(error, errorCapacity,
                           "cannot create the frame decoders");
  }
  for (size_t i = 0; i < threads; ++i) {
    workers[i] = (PassThread){pass, &gpu->workers[i], 0};
    workers[i].threaded =
        i != 0 && pthread_create(&handles[i], NULL, run_pass, &workers[i]) == 0;
  }
  run_pass(&workers[0]);
  for (size_t i = 1; i < threads; ++i) {
    if (workers[i].threaded)
      pthread_join(handles[i], NULL);
    else
      run_pass(&workers[i]);
  }
  free(handles);
  free(workers);
  if (atomic_load(&pass->failed))
    return geozl_gpu_error(error, errorCapacity, "frame %zu: %s",
                           pass->errorFrame, pass->error);
  return 0;
}

typedef struct {
  GeozlGpuMemory *memory;
  void *host;
  size_t offset;
  size_t bytes;
  PlaceCopy *copies;
  size_t copyCount;
  uint32_t maxRows;
} PlaceSource;

static int queue(GeozlGpu *gpu, GeozlGpuBatch *batch,
                 const GeozlGpuDispatch dispatches[], size_t dispatchCount,
                 void *stream, char *error, size_t errorCapacity) {
  if (batch == NULL)
    return 1;
  if (gpu->pendingCount == MAX_PENDING) {
    geozl_gpu_batch_destroy(batch);
    return geozl_gpu_error(error, errorCapacity, "too many GPU batches");
  }
  gpu->pending[gpu->pendingCount++] = batch;
  return geozl_gpu_batch_enqueue(batch, dispatches, dispatchCount, stream,
                                 error, errorCapacity);
}

static int ensure_scratch(GeozlGpu *gpu, size_t bytes, char *error,
                          size_t errorCapacity) {
  if (gpu->scratch != NULL && gpu->scratch->bytes >= bytes)
    return 0;
  geozl_gpu_memory_destroy(gpu->scratch);
  const size_t grown =
      bytes <= SIZE_MAX - bytes / 4 ? bytes + bytes / 4 : bytes;
  gpu->scratch =
      geozl_gpu_memory_create(gpu->place[0], grown, error, errorCapacity);
  return gpu->scratch == NULL;
}

// Splits copies taller than one grid can cover and converts them to samples.
static int add_place_copies(PlaceSource *source, const GeozlGpuCopy *copy,
                            size_t frameBase, size_t sampleBytes) {
  for (size_t row = 0; row < copy->rows; row += PLACE_MAX_ROWS) {
    const uint32_t rows = (uint32_t)(
        copy->rows - row < PLACE_MAX_ROWS ? copy->rows - row : PLACE_MAX_ROWS);
    const size_t from = frameBase + copy->source + row * copy->sourceRow;
    const size_t to = copy->destination + row * copy->destinationRow;
    const size_t fromEnd = span_end(from, rows, copy->sourceRow, copy->width,
                                    copy->sourceStep, sampleBytes);
    const size_t toEnd = span_end(to, rows, copy->destinationRow, copy->width,
                                  copy->destinationStep, sampleBytes);
    if (fromEnd / sampleBytes > UINT32_MAX || toEnd / sampleBytes > UINT32_MAX)
      return 1;
    source->copies[source->copyCount++] = (PlaceCopy){
        (uint32_t)(from / sampleBytes),
        rows > 1 ? (uint32_t)(copy->sourceRow / sampleBytes) : 0,
        copy->width > 1 ? (uint32_t)(copy->sourceStep / sampleBytes) : 0,
        (uint32_t)(to / sampleBytes),
        rows > 1 ? (uint32_t)(copy->destinationRow / sampleBytes) : 0,
        copy->width > 1 ? (uint32_t)(copy->destinationStep / sampleBytes) : 0,
        copy->width,
        rows,
    };
    if (rows > source->maxRows)
      source->maxRows = rows;
  }
  return 0;
}

static size_t piece_count(const GeozlGpuCopy *copy) {
  return ((size_t)copy->rows + PLACE_MAX_ROWS - 1) / PLACE_MAX_ROWS;
}

int geozl_gpu_decode(GeozlGpu *gpu, const GeozlGpuFrame frames[],
                     size_t frameCount, const GeozlGpuCopy copies[],
                     size_t copyCount, size_t sampleBytes, size_t elementBytes,
                     const GeozlGpuTarget *target, char *error,
                     size_t errorCapacity) {
  if (error != NULL && errorCapacity != 0)
    error[0] = '\0';
  if (gpu == NULL)
    return geozl_gpu_error(error, errorCapacity, "invalid GPU decoder");
  if (finish_pending(gpu, error, errorCapacity) != 0)
    return 1;
  if (frames == NULL || frameCount == 0 || frameCount > UINT32_MAX ||
      (copies == NULL && copyCount != 0) || place_slot(sampleBytes) < 0 ||
      elementBytes == 0 || sampleBytes % elementBytes != 0 || target == NULL ||
      target->memory == NULL || target->bytes == 0)
    return geozl_gpu_error(error, errorCapacity, "invalid GPU decode input");
  int aligned = target->offset % sampleBytes == 0;
  if (gpu->backend == GEOZL_GPU_BACKEND_CUDA) {
    const uintptr_t memory = (uintptr_t)target->memory;
    aligned = target->offset <= UINTPTR_MAX - memory &&
              (memory + target->offset) % sampleBytes == 0;
  }
  if (!aligned)
    return geozl_gpu_error(error, errorCapacity, "unaligned GPU target");
  size_t decodedBytes = 0;
  for (size_t i = 0; i < frameCount; ++i) {
    if (frames[i].data == NULL || frames[i].size == 0 ||
        frames[i].decodedBytes == 0 ||
        frames[i].decodedBytes % sampleBytes != 0)
      return geozl_gpu_error(error, errorCapacity, "invalid frame %zu", i);
    if (add_size(&decodedBytes, frames[i].decodedBytes) ||
        decodedBytes > gpu->maxDecodedBytes)
      return geozl_gpu_error(error, errorCapacity,
                             "decoded output exceeds the batch limit");
  }
  const size_t targetBytes = target->bytes / sampleBytes * sampleBytes;
  if (!copies_valid(frames, frameCount, copies, copyCount, sampleBytes,
                    targetBytes))
    return geozl_gpu_error(error, errorCapacity,
                           "a copy reaches outside its frame or the target");
  if (copyCount == 0)
    return 0;

  GeozlGpuProgram *place =
      place_program(gpu, sampleBytes, error, errorCapacity);
  if (place == NULL)
    return 1;
  const int device = elementBytes == sampleBytes && sampleBytes <= 2;
  GeozlGpuArena *staging = &gpu->staging;
  geozl_gpu_arena_reset(staging);

  int failed = 1;
  PforGpuWork pforWork = {0};
  PivcoGpuWork pivcoWork = {0};
  PlaceSource sources[SOURCES] = {0};
  uint8_t *routes = calloc(frameCount, sizeof(*routes));
  size_t *slots = calloc(frameCount, sizeof(*slots));
  size_t *bases = calloc(frameCount, sizeof(*bases));
  size_t *ids = calloc(frameCount, sizeof(*ids));
  CapturedPforFrame *pforCaptured = NULL;
  CapturedFrame *pivcoCaptured = NULL;
  PforGpuFrame *pforFrames = NULL;
  GeozlGpuFrameShape *shapes = NULL;
  Pass pass = {0};
  int locked = 0;
  if (routes == NULL || slots == NULL || bases == NULL || ids == NULL) {
    geozl_gpu_error(error, errorCapacity, "out of memory");
    goto done;
  }

  // Slots are laid out per route before any frame is touched, so threads
  // capture and decode into disjoint staging. Frames past the 32-bit budget of
  // the device kernels decode on the host; a PivCo frame also pads each of its
  // byte lanes to 256.
  size_t pforBytes = 0, pivcoBytes = 0, hostBytes = 0;
  size_t pforCount = 0, pivcoCount = 0, deviceOutput = 0;
  for (size_t i = 0; i < frameCount; ++i) {
    routes[i] = device ? (uint8_t)route_of(&frames[i]) : ROUTE_HOST;
    if (routes[i] != ROUTE_HOST) {
      const size_t payload = routes[i] == ROUTE_PFOR ? pforBytes : pivcoBytes;
      const size_t output = frames[i].decodedBytes <= DEVICE_BUDGET
                                ? frames[i].decodedBytes + 256 * sampleBytes
                                : SIZE_MAX;
      const size_t slot = round_up(frames[i].size, 16);
      if (slot > DEVICE_BUDGET - payload ||
          output > DEVICE_BUDGET - deviceOutput)
        routes[i] = ROUTE_HOST;
      else
        deviceOutput += output;
    }
    size_t *total = routes[i] == ROUTE_PFOR    ? &pforBytes
                    : routes[i] == ROUTE_PIVCO ? &pivcoBytes
                                               : &hostBytes;
    slots[i] = *total;
    if (add_size(total, routes[i] == ROUTE_HOST
                            ? frames[i].decodedBytes
                            : round_up(frames[i].size, 16))) {
      geozl_gpu_error(error, errorCapacity, "GPU batch is too large");
      goto done;
    }
    pforCount += routes[i] == ROUTE_PFOR;
    pivcoCount += routes[i] == ROUTE_PIVCO;
  }
  uint8_t *pforPayload =
      pforCount == 0 ? NULL
                     : geozl_gpu_arena_alloc(
                           staging, pforBytes + GEOZL_GPU_PAYLOAD_PADDING);
  uint8_t *pivcoPayload =
      pivcoCount == 0 ? NULL
                      : geozl_gpu_arena_alloc(
                            staging, pivcoBytes + GEOZL_GPU_PAYLOAD_PADDING);
  uint8_t *host =
      hostBytes == 0 ? NULL : geozl_gpu_arena_alloc(staging, hostBytes);
  pforCaptured =
      pforCount == 0 ? NULL : calloc(frameCount, sizeof(*pforCaptured));
  pivcoCaptured =
      pivcoCount == 0 ? NULL : calloc(frameCount, sizeof(*pivcoCaptured));
  if ((pforCount != 0 && (pforPayload == NULL || pforCaptured == NULL)) ||
      (pivcoCount != 0 && (pivcoPayload == NULL || pivcoCaptured == NULL)) ||
      (hostBytes != 0 && host == NULL)) {
    geozl_gpu_error(error, errorCapacity, "%s", geozl_gpu_arena_error(staging));
    goto done;
  }

  pass = (Pass){
      .gpu = gpu,
      .frames = frames,
      .frameCount = frameCount,
      .routes = routes,
      .slots = slots,
      .pforPayload = pforPayload,
      .pivcoPayload = pivcoPayload,
      .host = host,
      .pforCaptured = pforCaptured,
      .pivcoCaptured = pivcoCaptured,
      .elementBytes = elementBytes,
      .sampleBytes = sampleBytes,
  };
  if (pthread_mutex_init(&pass.lock, NULL) != 0) {
    geozl_gpu_error(error, errorCapacity, "cannot create a lock");
    goto done;
  }
  locked = 1;
  if (run_frames(gpu, &pass, error, errorCapacity) != 0)
    goto done;

  // Captured frames get their place in the device output in frame order. A
  // header that disagrees with the frame also sends it to the host.
  pforFrames = pforCount == 0 ? NULL : calloc(pforCount, sizeof(*pforFrames));
  shapes = pivcoCount == 0 ? NULL : calloc(pivcoCount, sizeof(*shapes));
  if ((pforCount != 0 && pforFrames == NULL) ||
      (pivcoCount != 0 && shapes == NULL)) {
    geozl_gpu_error(error, errorCapacity, "out of memory");
    goto done;
  }
  size_t pforOutput = 0, pivcoOutput = 0, fallbackBytes = 0;
  size_t pforUsed = 0, pivcoUsed = 0;
  size_t *pforIds = ids;
  size_t *pivcoIds = ids + pforCount;
  for (size_t i = 0; i < frameCount; ++i) {
    if (routes[i] == ROUTE_PFOR) {
      if (pfor_gpu_frame_init(&pforFrames[pforUsed], &pforCaptured[i], slots[i],
                              sampleBytes, frames[i].decodedBytes,
                              pforOutput / sampleBytes) == 0) {
        bases[i] = pforOutput;
        pforIds[pforUsed++] = i;
        pforOutput += frames[i].decodedBytes;
        continue;
      }
      routes[i] = ROUTE_FALLBACK;
    } else if (routes[i] == ROUTE_PIVCO) {
      if (pivco_gpu_frame_init(&pivcoCaptured[i], &shapes[pivcoUsed], slots[i],
                               sampleBytes, frames[i].decodedBytes,
                               pivcoOutput / sampleBytes) == 0) {
        if (pivcoUsed != i)
          pivcoCaptured[pivcoUsed] = pivcoCaptured[i];
        bases[i] = pivcoOutput;
        pivcoIds[pivcoUsed++] = i;
        pivcoOutput += frames[i].decodedBytes;
        continue;
      }
      routes[i] = ROUTE_FALLBACK;
    }
    if (routes[i] == ROUTE_HOST) {
      bases[i] = slots[i];
    } else if (routes[i] == ROUTE_FALLBACK) {
      bases[i] = fallbackBytes;
      fallbackBytes += frames[i].decodedBytes;
    }
  }

  uint8_t *fallback =
      fallbackBytes == 0 ? NULL : geozl_gpu_arena_alloc(staging, fallbackBytes);
  if (fallbackBytes != 0 && fallback == NULL) {
    geozl_gpu_error(error, errorCapacity, "%s", geozl_gpu_arena_error(staging));
    goto done;
  }
  for (size_t i = 0; i < frameCount; ++i) {
    char message[224];
    if (routes[i] == ROUTE_FALLBACK &&
        host_decode(&gpu->workers[0], &frames[i], fallback + bases[i],
                    elementBytes, sampleBytes, message, sizeof(message)) != 0) {
      geozl_gpu_error(error, errorCapacity, "frame %zu: %s", i, message);
      goto done;
    }
  }

  GeozlPforGpuDecoder *pfor = NULL;
  GeozlPivcoGpuDecoder *pivco = NULL;
  if ((pforUsed != 0 &&
       (pfor = pfor_decoder(gpu, sampleBytes, error, errorCapacity)) == NULL) ||
      (pivcoUsed != 0 &&
       (pivco = pivco_decoder(gpu, sampleBytes, error, errorCapacity)) == NULL))
    goto done;
  if (pforUsed != 0 &&
      pfor_gpu_work_build(&pforWork, staging, sampleBytes, pforFrames, pforUsed,
                          pforIds, pforPayload, pforBytes, NULL, 0, NULL, 0,
                          pforOutput, gpu->threads, error, errorCapacity))
    goto done;
  if (pivcoUsed != 0 &&
      pivco_gpu_work_build(&pivcoWork, staging, sampleBytes, pivcoCaptured,
                           shapes, pivcoUsed, pivcoIds, pivcoPayload,
                           pivcoBytes, NULL, 0, NULL, 0, pivcoOutput,
                           gpu->threads, error, errorCapacity))
    goto done;

  // Device-decoded frames share one scratch, PFOR first.
  const size_t pivcoOffset = round_up(pforOutput, GEOZL_GPU_STAGING_ALIGNMENT);
  const size_t deviceBytes =
      pivcoUsed == 0 ? pforOutput : pivcoOffset + pivcoWork.resultBytes;
  if (deviceBytes != 0 &&
      ensure_scratch(gpu, deviceBytes, error, errorCapacity) != 0)
    goto done;
  for (size_t i = 0; i < frameCount; ++i) {
    if (routes[i] == ROUTE_PIVCO)
      bases[i] += pivcoOffset;
  }

  size_t pieces[SOURCES] = {0};
  for (size_t i = 0; i < copyCount; ++i) {
    const int route = routes[copies[i].frame];
    const int source = route == ROUTE_HOST       ? SOURCE_HOST
                       : route == ROUTE_FALLBACK ? SOURCE_FALLBACK
                                                 : SOURCE_DEVICE;
    if (add_size(&pieces[source], piece_count(&copies[i]))) {
      geozl_gpu_error(error, errorCapacity, "GPU batch is too large");
      goto done;
    }
  }
  sources[SOURCE_DEVICE] =
      (PlaceSource){.memory = gpu->scratch, .bytes = deviceBytes};
  sources[SOURCE_HOST] = (PlaceSource){.host = host, .bytes = hostBytes};
  sources[SOURCE_FALLBACK] =
      (PlaceSource){.host = fallback, .bytes = fallbackBytes};
  for (size_t s = 0; s < SOURCES; ++s) {
    if (pieces[s] == 0)
      continue;
    sources[s].copies =
        pieces[s] > SIZE_MAX / sizeof(PlaceCopy)
            ? NULL
            : geozl_gpu_arena_alloc(staging, pieces[s] * sizeof(PlaceCopy));
    if (sources[s].copies == NULL) {
      geozl_gpu_error(error, errorCapacity, "%s",
                      geozl_gpu_arena_error(staging));
      goto done;
    }
  }
  for (size_t i = 0; i < copyCount; ++i) {
    const int route = routes[copies[i].frame];
    const int source = route == ROUTE_HOST       ? SOURCE_HOST
                       : route == ROUTE_FALLBACK ? SOURCE_FALLBACK
                                                 : SOURCE_DEVICE;
    if (add_place_copies(&sources[source], &copies[i], bases[copies[i].frame],
                         sampleBytes) != 0) {
      geozl_gpu_error(error, errorCapacity, "GPU copy exceeds 2^32 samples");
      goto done;
    }
  }

  gpu->target =
      geozl_gpu_memory_wrap(place, target->memory, target->offset + targetBytes,
                            error, errorCapacity);
  if (gpu->target == NULL)
    goto done;
  if (pforUsed != 0) {
    pforWork.buffers[PFOR_GPU_OUTPUT] =
        (GeozlGpuBuffer){.bytes = pforOutput,
                         .elementBytes = sampleBytes,
                         .flags = GEOZL_GPU_EXTERNAL,
                         .alias = -1,
                         .memory = gpu->scratch};
    if (queue(gpu,
              geozl_gpu_batch_create(geozl_pfor_gpu_decoder_program(pfor),
                                     pforWork.buffers, PFOR_GPU_BUFFERS, error,
                                     errorCapacity),
              pforWork.dispatches, pforWork.dispatchCount, target->stream,
              error, errorCapacity) != 0)
      goto done;
  }
  if (pivcoUsed != 0) {
    pivcoWork.buffers[PIVCO_GPU_OUTPUT] =
        (GeozlGpuBuffer){.bytes = pivcoWork.resultBytes,
                         .elementBytes = sampleBytes,
                         .flags = GEOZL_GPU_EXTERNAL,
                         .alias = -1,
                         .memory = gpu->scratch,
                         .memoryOffset = pivcoOffset};
    if (queue(gpu,
              geozl_gpu_batch_create(geozl_pivco_gpu_decoder_program(pivco),
                                     pivcoWork.buffers, PIVCO_GPU_BUFFERS,
                                     error, errorCapacity),
              pivcoWork.dispatches, pivcoWork.dispatchCount, target->stream,
              error, errorCapacity) != 0)
      goto done;
  }
  for (size_t s = 0; s < SOURCES; ++s) {
    const PlaceSource *source = &sources[s];
    if (source->copyCount == 0)
      continue;
    const size_t sourceBytes = source->bytes / sampleBytes * sampleBytes;
    GeozlGpuBuffer buffers[3] = {
        source->memory != NULL
            ? (GeozlGpuBuffer){.bytes = sourceBytes,
                               .elementBytes = sampleBytes,
                               .flags = GEOZL_GPU_EXTERNAL,
                               .alias = -1,
                               .memory = source->memory}
            : geozl_gpu_arena_input(staging, source->host,
                                    sourceBytes / sampleBytes, sampleBytes),
        geozl_gpu_arena_input(staging, source->copies, source->copyCount,
                              sizeof(PlaceCopy)),
        {.bytes = targetBytes,
         .elementBytes = sampleBytes,
         .flags = GEOZL_GPU_EXTERNAL,
         .alias = -1,
         .memory = gpu->target,
         .memoryOffset = target->offset},
    };
    if (source->copyCount > UINT_MAX) {
      geozl_gpu_error(error, errorCapacity, "too many GPU copies");
      goto done;
    }
    const GeozlGpuDispatch dispatch = {0, (unsigned)source->copyCount,
                                       (source->maxRows + 3) / 4, 128};
    if (queue(gpu,
              geozl_gpu_batch_create(place, buffers, 3, error, errorCapacity),
              &dispatch, 1, target->stream, error, errorCapacity) != 0)
      goto done;
  }
  failed = 0;

done:
  // A failed call still waits for the work it queued before returning.
  if (failed)
    finish_pending(gpu, NULL, 0);
  if (locked)
    pthread_mutex_destroy(&pass.lock);
  pivco_gpu_work_free(&pivcoWork);
  pfor_gpu_work_free(&pforWork);
  free(shapes);
  free(pforFrames);
  free(pivcoCaptured);
  free(pforCaptured);
  free(ids);
  free(bases);
  free(slots);
  free(routes);
  return failed;
}
