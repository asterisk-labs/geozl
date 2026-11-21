// Runs the public decoder over a backend that records batches instead of
// running kernels, so its deferred refusals, failed and invalid calls and
// overlapping calls are checked without a device. Both device types use the
// mock, and decode.c keeps CUDA batches queued until a later call.

#include "geozl/geozl.h"
#include "geozl/gpu.h"

#include "gpu/modules.h"
#include "gpu/pfor_decode.h"
#include "gpu/runtime_internal.h"

#include "openzl/codecs/zl_store.h"
#include "openzl/zl_compress.h"
#include "openzl/zl_compressor.h"
#include "openzl/zl_version.h"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ROWS 32
#define WIDTH 32
#define FRAME_BYTES (ROWS * WIDTH * 2)

#define REFUSED "frame 0: invalid PFOR block header"
#define EARLIER "in an earlier call, " REFUSED
#define INVALID "invalid GPU decode input"
#define PLACE_FAILED "mock place batch failed"
#define BUSY "GPU decoder is serving another call"
#define OUTSIDE "a copy reaches outside its frame or the target"

enum { PLACE, PFOR, PIVCO };

typedef struct {
  GeozlGpuProgram base;
  int kind;
} MockProgram;

typedef struct {
  GeozlGpuMemory base;
  void *data;
  int owned;
} MockMemory;

// refuse makes PFOR batches report frame 0 and failPlace fails place batches.
// hold parks the next PFOR batch until the test sets released.
static int refuse, failPlace, hold, held, released;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t moved = PTHREAD_COND_INITIALIZER;
static int failures;

static GeozlGpuProgram *program_create(const void *module, size_t moduleSize,
                                       int device, const char *const kernels[],
                                       size_t kernelCount, char *error,
                                       size_t errorCapacity) {
  (void)module;
  (void)moduleSize;
  (void)device;
  (void)kernelCount;
  MockProgram *program = calloc(1, sizeof(*program));
  if (program == NULL) {
    geozl_gpu_error(error, errorCapacity, "out of memory");
    return NULL;
  }
  program->kind = strcmp(kernels[0], "place") == 0           ? PLACE
                  : strcmp(kernels[0], "unpack_blocks") == 0 ? PFOR
                                                             : PIVCO;
  return &program->base;
}

static void program_destroy(GeozlGpuProgram *program) { free(program); }

static const char *program_device(const GeozlGpuProgram *program) {
  (void)program;
  return "mock";
}

static GeozlGpuStaging *staging_create(GeozlGpuProgram *program, size_t bytes,
                                       char *error, size_t errorCapacity) {
  (void)program;
  const size_t align = GEOZL_GPU_STAGING_ALIGNMENT;
  GeozlGpuStaging *staging = calloc(1, sizeof(*staging));
  if (staging != NULL)
    staging->data = aligned_alloc(align, (bytes + align - 1) / align * align);
  if (staging == NULL || staging->data == NULL) {
    free(staging);
    geozl_gpu_error(error, errorCapacity, "out of memory");
    return NULL;
  }
  return staging;
}

static void staging_destroy(GeozlGpuStaging *staging) {
  free(staging->data);
  free(staging);
}

static GeozlGpuBatch *batch_create(GeozlGpuProgram *program,
                                   const GeozlGpuBuffer buffers[],
                                   size_t bufferCount, char *error,
                                   size_t errorCapacity) {
  if (failPlace && ((MockProgram *)program)->kind == PLACE) {
    geozl_gpu_error(error, errorCapacity, PLACE_FAILED);
    return NULL;
  }
  GeozlGpuBatch *batch = calloc(1, sizeof(*batch));
  if (batch == NULL) {
    geozl_gpu_error(error, errorCapacity, "out of memory");
    return NULL;
  }
  geozl_gpu_batch_init(batch, program, buffers, bufferCount);
  return batch;
}

static void batch_destroy(GeozlGpuBatch *batch) { free(batch); }

static int dispatches_fit(const GeozlGpuProgram *program,
                          const GeozlGpuDispatch dispatches[],
                          size_t dispatchCount, char *error,
                          size_t errorCapacity) {
  (void)program;
  (void)dispatches;
  (void)dispatchCount;
  (void)error;
  (void)errorCapacity;
  return 0;
}

static int batch_run(GeozlGpuBatch *batch, const GeozlGpuDispatch dispatches[],
                     size_t dispatchCount, GeozlGpuRunTimes *times, char *error,
                     size_t errorCapacity) {
  (void)batch;
  (void)dispatches;
  (void)dispatchCount;
  (void)times;
  (void)error;
  (void)errorCapacity;
  return 0;
}

// A refused PFOR batch writes the staged status as its kernel would.
static int batch_enqueue(GeozlGpuBatch *batch,
                         const GeozlGpuDispatch dispatches[],
                         size_t dispatchCount, void *stream, char *error,
                         size_t errorCapacity) {
  (void)dispatches;
  (void)dispatchCount;
  (void)stream;
  (void)error;
  (void)errorCapacity;
  if (((MockProgram *)batch->program)->kind != PFOR)
    return 0;
  pthread_mutex_lock(&lock);
  if (hold) {
    hold = 0;
    held = 1;
    pthread_cond_broadcast(&moved);
    while (!released)
      pthread_cond_wait(&moved, &lock);
  }
  pthread_mutex_unlock(&lock);
  if (refuse) {
    uint32_t *status = batch->buffers[PFOR_GPU_STATUS].host;
    status[PFOR_GPU_STATUS_HEADER] = 0;
  }
  return 0;
}

static int batch_wait(GeozlGpuBatch *batch, char *error, size_t errorCapacity) {
  (void)batch;
  (void)error;
  (void)errorCapacity;
  return 0;
}

static int batch_download(GeozlGpuBatch *batch, char *error,
                          size_t errorCapacity) {
  (void)batch;
  (void)error;
  (void)errorCapacity;
  return 0;
}

static GeozlGpuMemory *batch_take(GeozlGpuBatch *batch, size_t buffer,
                                  char *error, size_t errorCapacity) {
  (void)batch;
  (void)buffer;
  geozl_gpu_error(error, errorCapacity, "the mock keeps no results");
  return NULL;
}

static GeozlGpuMemory *new_memory(void *data, int owned, char *error,
                                  size_t errorCapacity) {
  MockMemory *memory = data == NULL ? NULL : calloc(1, sizeof(*memory));
  if (memory == NULL) {
    if (owned)
      free(data);
    geozl_gpu_error(error, errorCapacity, "out of memory");
    return NULL;
  }
  memory->data = data;
  memory->owned = owned;
  return &memory->base;
}

static GeozlGpuMemory *memory_create(GeozlGpuProgram *program, size_t bytes,
                                     char *error, size_t errorCapacity) {
  (void)program;
  return new_memory(calloc(1, bytes), 1, error, errorCapacity);
}

static GeozlGpuMemory *memory_wrap(GeozlGpuProgram *program, void *data,
                                   size_t bytes, char *error,
                                   size_t errorCapacity) {
  (void)program;
  (void)bytes;
  return new_memory(data, 0, error, errorCapacity);
}

static void memory_destroy(GeozlGpuMemory *base) {
  MockMemory *memory = (MockMemory *)base;
  if (memory->owned)
    free(memory->data);
  free(memory);
}

static void *memory_data(const GeozlGpuMemory *memory) {
  return ((const MockMemory *)memory)->data;
}

static int memory_download(const GeozlGpuMemory *memory, void *dst,
                           size_t bytes, char *error, size_t errorCapacity) {
  (void)error;
  (void)errorCapacity;
  memcpy(dst, ((const MockMemory *)memory)->data, bytes);
  return 0;
}

#define MOCK_OPS(backend)                                                      \
  {                                                                            \
    .name = backend, .program_create = program_create,                         \
    .program_destroy = program_destroy, .program_device = program_device,      \
    .staging_create = staging_create, .staging_destroy = staging_destroy,      \
    .batch_create = batch_create, .batch_destroy = batch_destroy,              \
    .dispatches_fit = dispatches_fit, .batch_run = batch_run,                  \
    .batch_enqueue = batch_enqueue, .batch_wait = batch_wait,                  \
    .batch_download = batch_download, .batch_take = batch_take,                \
    .memory_create = memory_create, .memory_wrap = memory_wrap,                \
    .memory_destroy = memory_destroy, .memory_data = memory_data,              \
    .memory_download = memory_download,                                        \
  }

const GeozlGpuBackendOps geozl_gpu_cuda_ops = MOCK_OPS("cuda");
const GeozlGpuBackendOps geozl_gpu_metal_ops = MOCK_OPS("metal");

static const unsigned char module[1] = {0};
const GeozlGpuModule geozl_gpu_modules[] = {
    {"place1", module, 1}, {"place2", module, 1},  {"place4", module, 1},
    {"place8", module, 1}, {"place16", module, 1}, {"pfor1", module, 1},
    {"pfor2", module, 1},  {"pivco1", module, 1},  {"pivco2", module, 1},
};
const size_t geozl_gpu_module_count =
    sizeof(geozl_gpu_modules) / sizeof(geozl_gpu_modules[0]);

static uint8_t data[2][2 * FRAME_BYTES];
static GeozlGpuFrame frames[2];
static GeozlGpuCopy copies[2];

// A stored id>pfor frame, which routes to the device.
static size_t pfor_frame(uint8_t *dst, size_t capacity, const uint16_t *src,
                         size_t count) {
  size_t written = 0;
  ZL_Compressor *compressor = ZL_Compressor_create();
  ZL_CCtx *context = ZL_CCtx_create();
  if (compressor != NULL && context != NULL) {
    const ZL_GraphID graph = ZL_Compressor_registerStaticGraph_fromNode(
        compressor, geozl_node_pfor(compressor), ZL_GRAPHLIST(ZL_GRAPH_STORE));
    ZL_TypedRef *input = ZL_TypedRef_createNumeric(src, 2, count);
    if (input != NULL && ZL_GraphID_isValid(graph) &&
        !ZL_isError(ZL_Compressor_selectStartingGraphID(compressor, graph)) &&
        !ZL_isError(ZL_CCtx_refCompressor(context, compressor)) &&
        !ZL_isError(ZL_CCtx_setParameter(context, ZL_CParam_formatVersion,
                                         ZL_MAX_FORMAT_VERSION))) {
      const ZL_Report report =
          ZL_CCtx_compressTypedRef(context, dst, capacity, input);
      if (!ZL_isError(report))
        written = ZL_validResult(report);
    }
    ZL_TypedRef_free(input);
  }
  ZL_CCtx_free(context);
  ZL_Compressor_free(compressor);
  return written;
}

typedef struct {
  const char *device;
  int deferred;
  GeozlGpu *gpu;
  GeozlGpuTarget target;
} Run;

// want is NULL for success, or the start of the expected message.
static void check(const Run *run, const char *step, int status,
                  const char *error, const char *want) {
  if (want == NULL ? status == 0
                   : status != 0 && strncmp(error, want, strlen(want)) == 0)
    return;
  fprintf(stderr, "test_gpu_decode_mock: %s, %s: returned %d, \"%s\"\n",
          run->device, step, status, error);
  ++failures;
}

// A sample size of 3 makes the arguments invalid.
static void decode_expect(const Run *run, const char *step, size_t sampleBytes,
                          const char *want) {
  char error[256] = "";
  const int status =
      geozl_gpu_decode(run->gpu, frames, 2, copies, 2, sampleBytes, 2,
                       &run->target, NULL, error, sizeof(error));
  check(run, step, status, error, want);
}

static void synchronize_expect(const Run *run, const char *step,
                               const char *want) {
  char error[256] = "";
  check(run, step, geozl_gpu_synchronize(run->gpu, error, sizeof(error)), error,
        want);
}

// CUDA reports a refusal from the call two later, Metal from the call itself,
// and either reports it once.
static void check_refusal(const Run *run) {
  refuse = 1;
  decode_expect(run, "a refused call", 2, run->deferred ? NULL : REFUSED);
  refuse = 0;
  decode_expect(run, "the next call", 2, NULL);
  decode_expect(run, "the call after", 2, run->deferred ? EARLIER : NULL);
  decode_expect(run, "a later call", 2, NULL);
  synchronize_expect(run, "synchronize", NULL);
}

// A call that fails after its own batch was refused leaves nothing behind.
static void check_failed_call(const Run *run) {
  refuse = failPlace = 1;
  decode_expect(run, "a refused call that fails", 2, PLACE_FAILED);
  refuse = failPlace = 0;
  decode_expect(run, "the call after a failure", 2, NULL);
  synchronize_expect(run, "synchronize after a failure", NULL);
}

// A failed call and an invalid one keep the refusal of the call before them.
static void check_kept_refusal(const Run *run) {
  refuse = 1;
  decode_expect(run, "a refused call", 2, run->deferred ? NULL : REFUSED);
  refuse = 0;
  failPlace = 1;
  decode_expect(run, "a failing call", 2, PLACE_FAILED);
  failPlace = 0;
  decode_expect(run, "an invalid call", 3, INVALID);
  decode_expect(run, "the next valid call", 2, run->deferred ? EARLIER : NULL);
  synchronize_expect(run, "synchronize", NULL);
}

// An invalid call neither waits on nor reports queued work.
static void check_invalid_call(const Run *run) {
  refuse = 1;
  decode_expect(run, "a refused call", 2, run->deferred ? NULL : REFUSED);
  refuse = 0;
  decode_expect(run, "a call after it", 2, NULL);
  decode_expect(run, "an invalid call", 3, INVALID);
  synchronize_expect(run, "synchronize", run->deferred ? EARLIER : NULL);
  synchronize_expect(run, "synchronize again", NULL);
}

// A copy past 2^32 target samples is an invalid argument, refused before the
// call takes an earlier refusal.
static void check_wide_target(const Run *run) {
  refuse = 1;
  decode_expect(run, "a refused call", 2, run->deferred ? NULL : REFUSED);
  refuse = 0;
  decode_expect(run, "a call after it", 2, NULL);
  GeozlGpuCopy wide[2] = {copies[0], copies[1]};
  wide[1].destination = (size_t)1 << 33;
  GeozlGpuTarget big = run->target;
  big.bytes = ((size_t)1 << 33) + 4096;
  char error[256] = "";
  check(run, "a copy past 2^32 samples",
        geozl_gpu_decode(run->gpu, frames, 2, wide, 2, 2, 2, &big, NULL, error,
                         sizeof(error)),
        error, OUTSIDE);
  synchronize_expect(run, "synchronize", run->deferred ? EARLIER : NULL);
}

typedef struct {
  const Run *run;
  int status;
  char error[256];
} HeldCall;

static void *decode_held(void *opaque) {
  HeldCall *call = opaque;
  call->status = geozl_gpu_decode(call->run->gpu, frames, 2, copies, 2, 2, 2,
                                  &call->run->target, NULL, call->error,
                                  sizeof(call->error));
  // A decode that ends before the mock parks it still wakes the test.
  pthread_mutex_lock(&lock);
  hold = 0;
  held = 1;
  pthread_cond_broadcast(&moved);
  pthread_mutex_unlock(&lock);
  return NULL;
}

// Calls that overlap a decode fail at once, while alloc and free still work.
static void check_overlap(const Run *run) {
  HeldCall call = {run, 1, ""};
  pthread_mutex_lock(&lock);
  hold = 1;
  held = released = 0;
  pthread_mutex_unlock(&lock);
  pthread_t thread;
  if (pthread_create(&thread, NULL, decode_held, &call) != 0) {
    fprintf(stderr, "test_gpu_decode_mock: cannot start a thread\n");
    ++failures;
    return;
  }
  pthread_mutex_lock(&lock);
  while (!held)
    pthread_cond_wait(&moved, &lock);
  pthread_mutex_unlock(&lock);
  decode_expect(run, "an overlapping decode", 2, BUSY);
  synchronize_expect(run, "an overlapping synchronize", BUSY);
  char error[256] = "";
  GeozlGpuMemory *memory = geozl_gpu_alloc(run->gpu, 64, error, sizeof(error));
  check(run, "an alloc during a decode", memory == NULL, error, NULL);
  geozl_gpu_free(memory);
  pthread_mutex_lock(&lock);
  released = 1;
  pthread_cond_broadcast(&moved);
  pthread_mutex_unlock(&lock);
  pthread_join(thread, NULL);
  check(run, "the held decode", call.status, call.error, NULL);
  decode_expect(run, "a decode after it", 2, NULL);
  synchronize_expect(run, "synchronize after it", NULL);
}

int main(void) {
  uint16_t source[ROWS * WIDTH];
  for (size_t i = 0; i < ROWS * WIDTH; ++i)
    source[i] = (uint16_t)(1000 + i % 37);
  // Each frame sends its left half into one half of the target, so the copies
  // go through place.
  for (size_t f = 0; f < 2; ++f) {
    const size_t size =
        pfor_frame(data[f], sizeof(data[f]), source, ROWS * WIDTH);
    if (size == 0) {
      fprintf(stderr, "test_gpu_decode_mock: cannot build a frame\n");
      return 1;
    }
    frames[f] = (GeozlGpuFrame){data[f], size, FRAME_BYTES};
    copies[f] = (GeozlGpuCopy){.frame = f,
                               .sourceRow = WIDTH * 2,
                               .sourceStep = 2,
                               .destination = f * WIDTH,
                               .destinationRow = WIDTH * 2,
                               .destinationStep = 2,
                               .width = WIDTH / 2,
                               .rows = ROWS};
  }

  static const struct {
    int type;
    const char *name;
    int deferred;
  } devices[] = {{GEOZL_GPU_DEVICE_CUDA, "cuda", 1},
                 {GEOZL_GPU_DEVICE_METAL, "metal", 0}};
  for (size_t d = 0; d < sizeof(devices) / sizeof(devices[0]); ++d) {
    char error[256] = "";
    Run run = {devices[d].name,
               devices[d].deferred,
               geozl_gpu_create(devices[d].type, 0, 2, 1 << 20, 0, error,
                                sizeof(error)),
               {0}};
    GeozlGpuMemory *memory =
        run.gpu == NULL
            ? NULL
            : geozl_gpu_alloc(run.gpu, FRAME_BYTES, error, sizeof(error));
    if (memory == NULL) {
      fprintf(stderr, "test_gpu_decode_mock: %s: %s\n", run.device, error);
      geozl_gpu_destroy(run.gpu);
      return 1;
    }
    run.target =
        (GeozlGpuTarget){geozl_gpu_pointer(memory), 0, FRAME_BYTES, NULL};
    check_refusal(&run);
    check_failed_call(&run);
    check_kept_refusal(&run);
    check_invalid_call(&run);
    check_wide_target(&run);
    check_overlap(&run);
    geozl_gpu_free(memory);
    geozl_gpu_destroy(run.gpu);
  }
  if (failures != 0)
    return 1;
  puts("test_gpu_decode_mock: ok");
  return 0;
}
