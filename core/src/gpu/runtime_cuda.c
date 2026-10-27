#include "runtime_internal.h"

#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef int CUresult;
typedef int CUdevice;
typedef unsigned long long CUdeviceptr;
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;
typedef struct CUevent_st *CUevent;
typedef int CUjit_option;

#define CUDA_SUCCESS 0
#define CU_STREAM_NON_BLOCKING 1
#define CU_EVENT_DEFAULT 0
#define CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK 1
#define CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X 5
#define CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y 6
#define CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK 0
#define CU_JIT_ERROR_LOG_BUFFER 5
#define CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES 6

#define CU_API(X)                                                              \
  X(cuInit, "cuInit", CUresult, (unsigned))                                    \
  X(cuGetErrorString, "cuGetErrorString", CUresult, (CUresult, const char **)) \
  X(cuDeviceGet, "cuDeviceGet", CUresult, (CUdevice *, int))                   \
  X(cuDeviceGetName, "cuDeviceGetName", CUresult, (char *, int, CUdevice))     \
  X(cuDeviceGetAttribute, "cuDeviceGetAttribute", CUresult,                    \
    (int *, int, CUdevice))                                                    \
  X(cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain", CUresult,            \
    (CUcontext *, CUdevice))                                                   \
  X(cuDevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease_v2", CUresult,       \
    (CUdevice))                                                                \
  X(cuCtxPushCurrent, "cuCtxPushCurrent_v2", CUresult, (CUcontext))            \
  X(cuCtxPopCurrent, "cuCtxPopCurrent_v2", CUresult, (CUcontext *))            \
  X(cuModuleLoadDataEx, "cuModuleLoadDataEx", CUresult,                        \
    (CUmodule *, const void *, unsigned, CUjit_option *, void **))             \
  X(cuModuleUnload, "cuModuleUnload", CUresult, (CUmodule))                    \
  X(cuModuleGetFunction, "cuModuleGetFunction", CUresult,                      \
    (CUfunction *, CUmodule, const char *))                                    \
  X(cuModuleGetGlobal, "cuModuleGetGlobal_v2", CUresult,                       \
    (CUdeviceptr *, size_t *, CUmodule, const char *))                         \
  X(cuFuncGetAttribute, "cuFuncGetAttribute", CUresult,                        \
    (int *, int, CUfunction))                                                  \
  X(cuMemHostAlloc, "cuMemHostAlloc", CUresult, (void **, size_t, unsigned))   \
  X(cuMemFreeHost, "cuMemFreeHost", CUresult, (void *))                        \
  X(cuMemAlloc, "cuMemAlloc_v2", CUresult, (CUdeviceptr *, size_t))            \
  X(cuMemFree, "cuMemFree_v2", CUresult, (CUdeviceptr))                        \
  X(cuMemcpyHtoDAsync, "cuMemcpyHtoDAsync_v2", CUresult,                       \
    (CUdeviceptr, const void *, size_t, CUstream))                             \
  X(cuMemcpyDtoH, "cuMemcpyDtoH_v2", CUresult, (void *, CUdeviceptr, size_t))  \
  X(cuStreamCreate, "cuStreamCreate", CUresult, (CUstream *, unsigned))        \
  X(cuStreamDestroy, "cuStreamDestroy_v2", CUresult, (CUstream))               \
  X(cuStreamSynchronize, "cuStreamSynchronize", CUresult, (CUstream))          \
  X(cuEventCreate, "cuEventCreate", CUresult, (CUevent *, unsigned))           \
  X(cuEventDestroy, "cuEventDestroy_v2", CUresult, (CUevent))                  \
  X(cuEventRecord, "cuEventRecord", CUresult, (CUevent, CUstream))             \
  X(cuEventSynchronize, "cuEventSynchronize", CUresult, (CUevent))             \
  X(cuEventElapsedTime, "cuEventElapsedTime", CUresult,                        \
    (float *, CUevent, CUevent))                                               \
  X(cuLaunchKernel, "cuLaunchKernel", CUresult,                                \
    (CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,   \
     unsigned, CUstream, void **, void **))

#define CU_DECLARE(name, symbol, result, args) static result(*name) args;
CU_API(CU_DECLARE)

static void *cudaLibrary;
static pthread_mutex_t cudaLoadMutex = PTHREAD_MUTEX_INITIALIZER;

static int load_driver_locked(char *error, size_t errorCapacity) {
  if (cudaLibrary != NULL)
    return 0;
  void *library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  if (library == NULL)
    return geozl_gpu_error(error, errorCapacity, "no NVIDIA driver: %s",
                           dlerror());
#define CU_RESOLVE(name, symbol, result, args)                                 \
  do {                                                                         \
    void *address = dlsym(library, symbol);                                    \
    if (address == NULL) {                                                     \
      dlclose(library);                                                        \
      return geozl_gpu_error(error, errorCapacity, "libcuda lacks %s",         \
                             symbol);                                          \
    }                                                                          \
    memcpy(&name, &address, sizeof(address));                                  \
  } while (0);
  CU_API(CU_RESOLVE)
#undef CU_RESOLVE
  cudaLibrary = library;
  return 0;
}

static int load_driver(char *error, size_t errorCapacity) {
  if (pthread_mutex_lock(&cudaLoadMutex) != 0)
    return geozl_gpu_error(error, errorCapacity, "cannot lock the CUDA loader");
  const int failed = load_driver_locked(error, errorCapacity);
  pthread_mutex_unlock(&cudaLoadMutex);
  return failed;
}

static int cuda_ok(CUresult result, const char *operation, char *error,
                   size_t errorCapacity) {
  if (result == CUDA_SUCCESS)
    return 1;
  const char *message = NULL;
  if (cuGetErrorString != NULL)
    cuGetErrorString(result, &message);
  geozl_gpu_error(error, errorCapacity, "%s: %s", operation,
                  message == NULL ? "CUDA error" : message);
  return 0;
}

typedef struct {
  CUdeviceptr ptr;
  uint64_t count;
} SlangBuffer;

_Static_assert(sizeof(uintptr_t) >= sizeof(CUdeviceptr),
               "CUDA pointers require a 64-bit host");

typedef struct {
  GeozlGpuProgram base;
  CUdevice device;
  CUcontext context;
  CUmodule module;
  CUfunction kernels[GEOZL_GPU_MAX_KERNELS];
  CUdeviceptr params;
  size_t paramsSize;
  unsigned maxGridX;
  unsigned maxGridY;
  unsigned maxThreadsPerBlock;
  unsigned maxThreads[GEOZL_GPU_MAX_KERNELS];
  char deviceName[128];
} CudaProgram;

typedef struct {
  GeozlGpuBatch base;
  CUdeviceptr pointers[GEOZL_GPU_MAX_BUFFERS];
  size_t inputOffsets[GEOZL_GPU_MAX_BUFFERS];
  void *pinned;
  CUdeviceptr deviceInput;
  size_t inputBytes;
  SlangBuffer params[GEOZL_GPU_MAX_BUFFERS];
  CUstream stream;
  CUevent events[GEOZL_GPU_MAX_KERNELS + 2];
  size_t eventCount;
} CudaBatch;

typedef struct {
  GeozlGpuMemory base;
  CUdevice device;
  CUcontext context;
  CUdeviceptr pointer;
} CudaMemory;

typedef struct {
  GeozlGpuStaging base;
  CUdevice device;
  CUcontext context;
} CudaStaging;

static int push_context(CUcontext context, char *error, size_t errorCapacity) {
  return cuda_ok(cuCtxPushCurrent(context), "cuCtxPushCurrent", error,
                 errorCapacity);
}

static int pop_context(char *error, size_t errorCapacity) {
  CUcontext context;
  return cuda_ok(cuCtxPopCurrent(&context), "cuCtxPopCurrent", error,
                 errorCapacity);
}

static void cuda_program_destroy(GeozlGpuProgram *base) {
  CudaProgram *program = (CudaProgram *)base;
  if (program->context != NULL) {
    const int pushed = cuCtxPushCurrent(program->context) == CUDA_SUCCESS;
    if (program->module != NULL)
      cuModuleUnload(program->module);
    if (pushed) {
      CUcontext context;
      cuCtxPopCurrent(&context);
    }
    cuDevicePrimaryCtxRelease(program->device);
  }
  free(program);
}

// The JIT log names the PTX feature a too-old driver rejects.
static int load_module(CudaProgram *program, const void *module,
                       size_t moduleSize, char *error, size_t errorCapacity) {
  char *ptx = malloc(moduleSize + 1);
  if (ptx == NULL)
    return geozl_gpu_error(error, errorCapacity, "out of memory");
  memcpy(ptx, module, moduleSize);
  ptx[moduleSize] = 0;
  char log[512] = {0};
  CUjit_option options[] = {CU_JIT_ERROR_LOG_BUFFER,
                            CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
  void *values[] = {log, (void *)(uintptr_t)sizeof(log)};
  const CUresult result =
      cuModuleLoadDataEx(&program->module, ptx, 2, options, values);
  free(ptx);
  if (result == CUDA_SUCCESS)
    return 0;
  cuda_ok(result, "cuModuleLoadData", error, errorCapacity);
  log[strcspn(log, "\n")] = 0;
  if (log[0] != 0 && error != NULL && errorCapacity != 0) {
    const size_t used = strlen(error);
    snprintf(error + used, errorCapacity - used, ": %s", log);
  }
  program->module = NULL;
  return 1;
}

static GeozlGpuProgram *cuda_program_create(const void *module,
                                            size_t moduleSize, int deviceIndex,
                                            const char *const kernels[],
                                            size_t kernelCount, char *error,
                                            size_t errorCapacity) {
  if (load_driver(error, errorCapacity) != 0 ||
      !cuda_ok(cuInit(0), "cuInit", error, errorCapacity))
    return NULL;

  CudaProgram *program = calloc(1, sizeof(*program));
  if (program == NULL) {
    geozl_gpu_error(error, errorCapacity, "out of memory");
    return NULL;
  }
  int maxGridX = 0;
  int maxGridY = 0;
  int maxThreads = 0;
  if (!cuda_ok(cuDeviceGet(&program->device, deviceIndex), "cuDeviceGet", error,
               errorCapacity) ||
      !cuda_ok(cuDeviceGetName(program->deviceName,
                               (int)sizeof(program->deviceName),
                               program->device),
               "cuDeviceGetName", error, errorCapacity) ||
      !cuda_ok(cuDeviceGetAttribute(&maxGridX,
                                    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
                                    program->device),
               "cuDeviceGetAttribute", error, errorCapacity) ||
      !cuda_ok(cuDeviceGetAttribute(&maxGridY,
                                    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
                                    program->device),
               "cuDeviceGetAttribute", error, errorCapacity) ||
      !cuda_ok(cuDeviceGetAttribute(&maxThreads,
                                    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                    program->device),
               "cuDeviceGetAttribute", error, errorCapacity)) {
    cuda_program_destroy(&program->base);
    return NULL;
  }
  if (maxGridX <= 0 || maxGridY <= 0 || maxThreads <= 0) {
    geozl_gpu_error(error, errorCapacity,
                    "CUDA device reports invalid launch limits");
    cuda_program_destroy(&program->base);
    return NULL;
  }
  program->maxGridX = (unsigned)maxGridX;
  program->maxGridY = (unsigned)maxGridY;
  program->maxThreadsPerBlock = (unsigned)maxThreads;
  if (!cuda_ok(cuDevicePrimaryCtxRetain(&program->context, program->device),
               "cuDevicePrimaryCtxRetain", error, errorCapacity)) {
    cuda_program_destroy(&program->base);
    return NULL;
  }
  if (!push_context(program->context, error, errorCapacity)) {
    cuda_program_destroy(&program->base);
    return NULL;
  }
  if (load_module(program, module, moduleSize, error, errorCapacity) != 0) {
    pop_context(NULL, 0);
    cuda_program_destroy(&program->base);
    return NULL;
  }

  for (size_t i = 0; i < kernelCount; ++i) {
    int kernelMaxThreads = 0;
    if (!cuda_ok(cuModuleGetFunction(&program->kernels[i], program->module,
                                     kernels[i]),
                 "cuModuleGetFunction", error, errorCapacity) ||
        !cuda_ok(cuFuncGetAttribute(&kernelMaxThreads,
                                    CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                    program->kernels[i]),
                 "cuFuncGetAttribute", error, errorCapacity) ||
        kernelMaxThreads <= 0) {
      pop_context(NULL, 0);
      cuda_program_destroy(&program->base);
      return NULL;
    }
    program->maxThreads[i] = (unsigned)kernelMaxThreads;
  }
  if (!cuda_ok(cuModuleGetGlobal(&program->params, &program->paramsSize,
                                 program->module, "SLANG_globalParams"),
               "cuModuleGetGlobal", error, errorCapacity)) {
    pop_context(NULL, 0);
    cuda_program_destroy(&program->base);
    return NULL;
  }
  if (!pop_context(error, errorCapacity)) {
    cuda_program_destroy(&program->base);
    return NULL;
  }
  return &program->base;
}

static const char *cuda_program_device(const GeozlGpuProgram *program) {
  return ((const CudaProgram *)program)->deviceName;
}

static void cuda_staging_destroy(GeozlGpuStaging *base) {
  CudaStaging *staging = (CudaStaging *)base;
  if (staging->context != NULL) {
    if (cuCtxPushCurrent(staging->context) == CUDA_SUCCESS) {
      if (base->data != NULL)
        cuMemFreeHost(base->data);
      CUcontext context;
      cuCtxPopCurrent(&context);
    }
    cuDevicePrimaryCtxRelease(staging->device);
  }
  free(staging);
}

static GeozlGpuStaging *cuda_staging_create(GeozlGpuProgram *base, size_t bytes,
                                            char *error, size_t errorCapacity) {
  const CudaProgram *program = (const CudaProgram *)base;
  CudaStaging *staging = calloc(1, sizeof(*staging));
  if (staging == NULL) {
    geozl_gpu_error(error, errorCapacity, "out of memory");
    return NULL;
  }
  staging->device = program->device;
  if (!cuda_ok(cuDevicePrimaryCtxRetain(&staging->context, staging->device),
               "cuDevicePrimaryCtxRetain", error, errorCapacity)) {
    free(staging);
    return NULL;
  }
  if (!push_context(staging->context, error, errorCapacity)) {
    cuda_staging_destroy(&staging->base);
    return NULL;
  }
  const int failed = !cuda_ok(cuMemHostAlloc(&staging->base.data, bytes, 0),
                              "cuMemHostAlloc", error, errorCapacity);
  if (!pop_context(failed ? NULL : error, failed ? 0 : errorCapacity) ||
      failed) {
    cuda_staging_destroy(&staging->base);
    return NULL;
  }
  return &staging->base;
}

static void cuda_batch_destroy(GeozlGpuBatch *base) {
  CudaBatch *batch = (CudaBatch *)base;
  const CudaProgram *program = (const CudaProgram *)base->program;
  if (cuCtxPushCurrent(program->context) == CUDA_SUCCESS) {
    for (size_t i = 0; i < batch->eventCount; ++i)
      cuEventDestroy(batch->events[i]);
    if (batch->stream != NULL)
      cuStreamDestroy(batch->stream);
    for (size_t i = 0; i < base->bufferCount; ++i) {
      if (base->buffers[i].alias < 0 && batch->inputOffsets[i] == SIZE_MAX &&
          batch->pointers[i] != 0)
        cuMemFree(batch->pointers[i]);
    }
    if (batch->deviceInput != 0)
      cuMemFree(batch->deviceInput);
    if (batch->pinned != NULL)
      cuMemFreeHost(batch->pinned);
    CUcontext context;
    cuCtxPopCurrent(&context);
  }
  free(batch);
}

static GeozlGpuBatch *cuda_batch_create(GeozlGpuProgram *base,
                                        const GeozlGpuBuffer buffers[],
                                        size_t bufferCount, char *error,
                                        size_t errorCapacity) {
  CudaProgram *program = (CudaProgram *)base;
  if (program->paramsSize != bufferCount * sizeof(SlangBuffer)) {
    geozl_gpu_error(error, errorCapacity,
                    "CUDA module declares %zu bytes of buffers, need %zu",
                    program->paramsSize, bufferCount * sizeof(SlangBuffer));
    return NULL;
  }
  if (!push_context(program->context, error, errorCapacity))
    return NULL;

  CudaBatch *batch = calloc(1, sizeof(*batch));
  if (batch == NULL) {
    geozl_gpu_error(error, errorCapacity, "out of memory");
    pop_context(NULL, 0);
    return NULL;
  }
  geozl_gpu_batch_init(&batch->base, base, buffers, bufferCount);
  for (size_t i = 0; i < bufferCount; ++i)
    batch->inputOffsets[i] = SIZE_MAX;

  int unstaged = 0;
  for (size_t i = 0; i < bufferCount; ++i) {
    if (buffers[i].alias < 0 && (buffers[i].flags & GEOZL_GPU_COPY_IN) != 0) {
      unstaged |= buffers[i].staging == NULL;
      if (buffers[i].bytes > SIZE_MAX - 255) {
        geozl_gpu_error(error, errorCapacity, "CUDA input %zu is too large", i);
        goto fail;
      }
      const size_t padded = (buffers[i].bytes + 255) & ~(size_t)255;
      if (padded > SIZE_MAX - batch->inputBytes) {
        geozl_gpu_error(error, errorCapacity, "CUDA inputs are too large");
        goto fail;
      }
      batch->inputOffsets[i] = batch->inputBytes;
      batch->inputBytes += padded;
    }
  }
  // Staged inputs upload from their own pages; the rest pass through pinned
  // memory laid out like the device copy.
  if ((unstaged &&
       !cuda_ok(cuMemHostAlloc(&batch->pinned, batch->inputBytes, 0),
                "cuMemHostAlloc", error, errorCapacity)) ||
      (batch->inputBytes != 0 &&
       !cuda_ok(cuMemAlloc(&batch->deviceInput, batch->inputBytes),
                "cuMemAlloc", error, errorCapacity)))
    goto fail;

  for (size_t i = 0; i < bufferCount; ++i) {
    if (buffers[i].alias >= 0) {
      batch->pointers[i] = batch->pointers[buffers[i].alias];
    } else if (batch->inputOffsets[i] != SIZE_MAX) {
      batch->pointers[i] = batch->deviceInput + batch->inputOffsets[i];
    } else if (!cuda_ok(cuMemAlloc(&batch->pointers[i], buffers[i].bytes),
                        "cuMemAlloc", error, errorCapacity)) {
      goto fail;
    }
    batch->params[i] = (SlangBuffer){
        batch->pointers[i], buffers[i].bytes / buffers[i].elementBytes};
  }
  if (!cuda_ok(cuStreamCreate(&batch->stream, CU_STREAM_NON_BLOCKING),
               "cuStreamCreate", error, errorCapacity))
    goto fail;
  for (size_t i = 0; i < GEOZL_GPU_MAX_KERNELS + 2; ++i) {
    if (!cuda_ok(cuEventCreate(&batch->events[i], CU_EVENT_DEFAULT),
                 "cuEventCreate", error, errorCapacity))
      goto fail;
    ++batch->eventCount;
  }
  if (!pop_context(error, errorCapacity)) {
    cuda_batch_destroy(&batch->base);
    return NULL;
  }
  return &batch->base;

fail:
  cuda_batch_destroy(&batch->base);
  pop_context(NULL, 0);
  return NULL;
}

static int cuda_dispatches_fit(const GeozlGpuProgram *base,
                               const GeozlGpuDispatch dispatches[],
                               size_t dispatchCount, char *error,
                               size_t errorCapacity) {
  const CudaProgram *program = (const CudaProgram *)base;
  for (size_t i = 0; i < dispatchCount; ++i) {
    const GeozlGpuDispatch *dispatch = &dispatches[i];
    if (dispatch->groupsX > program->maxGridX ||
        dispatch->groupsY > program->maxGridY ||
        dispatch->threads > program->maxThreadsPerBlock ||
        dispatch->threads > program->maxThreads[dispatch->kernel])
      return geozl_gpu_error(error, errorCapacity,
                             "CUDA dispatch %zu exceeds device limits", i);
  }
  return 0;
}

// Kernels read their buffer pointers from the module-wide SLANG_globalParams,
// so the copy rides the batch stream ahead of the launches. The pageable source
// is staged before cuMemcpyHtoDAsync returns.
static int cuda_batch_run(GeozlGpuBatch *base,
                          const GeozlGpuDispatch dispatches[],
                          size_t dispatchCount, GeozlGpuRunTimes *times,
                          char *error, size_t errorCapacity) {
  CudaBatch *batch = (CudaBatch *)base;
  const CudaProgram *program = (const CudaProgram *)base->program;
  if (!push_context(program->context, error, errorCapacity))
    return 1;

  for (size_t i = 0; i < base->bufferCount; ++i) {
    const GeozlGpuBuffer *buffer = &base->buffers[i];
    if (batch->inputOffsets[i] != SIZE_MAX && buffer->staging == NULL)
      memcpy((uint8_t *)batch->pinned + batch->inputOffsets[i], buffer->host,
             buffer->bytes);
  }
  int failed = 1;
  if (!cuda_ok(cuEventRecord(batch->events[0], batch->stream), "cuEventRecord",
               error, errorCapacity) ||
      !cuda_ok(cuMemcpyHtoDAsync(program->params, batch->params,
                                 program->paramsSize, batch->stream),
               "cuMemcpyHtoDAsync", error, errorCapacity))
    goto done;
  for (size_t i = 0; i < base->bufferCount; ++i) {
    const GeozlGpuBuffer *buffer = &base->buffers[i];
    if (batch->inputOffsets[i] == SIZE_MAX)
      continue;
    const void *source =
        buffer->staging != NULL
            ? buffer->host
            : (const uint8_t *)batch->pinned + batch->inputOffsets[i];
    if (!cuda_ok(cuMemcpyHtoDAsync(batch->pointers[i], source, buffer->bytes,
                                   batch->stream),
                 "cuMemcpyHtoDAsync", error, errorCapacity))
      goto done;
  }
  if (!cuda_ok(cuEventRecord(batch->events[1], batch->stream), "cuEventRecord",
               error, errorCapacity))
    goto done;

  for (size_t i = 0; i < dispatchCount; ++i) {
    const GeozlGpuDispatch *dispatch = &dispatches[i];
    if (!cuda_ok(cuLaunchKernel(program->kernels[dispatch->kernel],
                                dispatch->groupsX, dispatch->groupsY, 1,
                                dispatch->threads, 1, 1, 0, batch->stream, NULL,
                                NULL),
                 "cuLaunchKernel", error, errorCapacity) ||
        !cuda_ok(cuEventRecord(batch->events[i + 2], batch->stream),
                 "cuEventRecord", error, errorCapacity))
      goto done;
  }
  if (!cuda_ok(cuEventSynchronize(batch->events[dispatchCount + 1]),
               "cuEventSynchronize", error, errorCapacity))
    goto done;

  float milliseconds = 0;
  if (!cuda_ok(
          cuEventElapsedTime(&milliseconds, batch->events[0], batch->events[1]),
          "cuEventElapsedTime", error, errorCapacity))
    goto done;
  times->upload = milliseconds / 1000.0;
  for (size_t i = 0; i < dispatchCount; ++i) {
    if (!cuda_ok(cuEventElapsedTime(&milliseconds, batch->events[i + 1],
                                    batch->events[i + 2]),
                 "cuEventElapsedTime", error, errorCapacity))
      goto done;
    times->dispatch[i] = milliseconds / 1000.0;
  }
  failed = 0;

done:
  // Work queued before a failure still reads host and device buffers.
  if (failed)
    cuStreamSynchronize(batch->stream);
  if (!pop_context(failed ? NULL : error, failed ? 0 : errorCapacity))
    return 1;
  return failed;
}

static int cuda_batch_download(GeozlGpuBatch *base, char *error,
                               size_t errorCapacity) {
  CudaBatch *batch = (CudaBatch *)base;
  if (!push_context(((const CudaProgram *)base->program)->context, error,
                    errorCapacity))
    return 1;
  int failed = 0;
  for (size_t i = 0; i < base->bufferCount; ++i) {
    const GeozlGpuBuffer *buffer = &base->buffers[i];
    if (buffer->alias < 0 && (buffer->flags & GEOZL_GPU_COPY_OUT) != 0 &&
        !cuda_ok(cuMemcpyDtoH(buffer->host, batch->pointers[i], buffer->bytes),
                 "cuMemcpyDtoH", error, errorCapacity)) {
      failed = 1;
      break;
    }
  }
  if (!pop_context(failed ? NULL : error, failed ? 0 : errorCapacity))
    return 1;
  return failed;
}

static GeozlGpuMemory *cuda_batch_take(GeozlGpuBatch *base, size_t bufferIndex,
                                       char *error, size_t errorCapacity) {
  CudaBatch *batch = (CudaBatch *)base;
  const CudaProgram *program = (const CudaProgram *)base->program;
  CudaMemory *memory = calloc(1, sizeof(*memory));
  if (memory == NULL) {
    geozl_gpu_error(error, errorCapacity, "out of memory");
    return NULL;
  }
  memory->device = program->device;
  if (!cuda_ok(cuDevicePrimaryCtxRetain(&memory->context, memory->device),
               "cuDevicePrimaryCtxRetain", error, errorCapacity)) {
    free(memory);
    return NULL;
  }
  memory->pointer = batch->pointers[bufferIndex];
  batch->pointers[bufferIndex] = 0;
  return &memory->base;
}

static void cuda_memory_destroy(GeozlGpuMemory *base) {
  CudaMemory *memory = (CudaMemory *)base;
  if (memory->context != NULL) {
    if (cuCtxPushCurrent(memory->context) == CUDA_SUCCESS) {
      if (memory->pointer != 0)
        cuMemFree(memory->pointer);
      CUcontext context;
      cuCtxPopCurrent(&context);
    }
    cuDevicePrimaryCtxRelease(memory->device);
  }
  free(memory);
}

static void *cuda_memory_data(const GeozlGpuMemory *memory) {
  return (void *)(uintptr_t)((const CudaMemory *)memory)->pointer;
}

static int cuda_memory_download(const GeozlGpuMemory *base, void *dst,
                                size_t bytes, char *error,
                                size_t errorCapacity) {
  const CudaMemory *memory = (const CudaMemory *)base;
  if (!push_context(memory->context, error, errorCapacity))
    return 1;
  const int failed = !cuda_ok(cuMemcpyDtoH(dst, memory->pointer, bytes),
                              "cuMemcpyDtoH", error, errorCapacity);
  if (!pop_context(failed ? NULL : error, failed ? 0 : errorCapacity))
    return 1;
  return failed;
}

const GeozlGpuBackendOps geozl_gpu_cuda_ops = {
    .name = "cuda",
    .program_create = cuda_program_create,
    .program_destroy = cuda_program_destroy,
    .program_device = cuda_program_device,
    .staging_create = cuda_staging_create,
    .staging_destroy = cuda_staging_destroy,
    .batch_create = cuda_batch_create,
    .batch_destroy = cuda_batch_destroy,
    .dispatches_fit = cuda_dispatches_fit,
    .batch_run = cuda_batch_run,
    .batch_download = cuda_batch_download,
    .batch_take = cuda_batch_take,
    .memory_destroy = cuda_memory_destroy,
    .memory_data = cuda_memory_data,
    .memory_download = cuda_memory_download,
};
