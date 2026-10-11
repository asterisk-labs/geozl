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

#define CUDA_SUCCESS 0
#define CU_STREAM_NON_BLOCKING 1
#define CU_EVENT_DEFAULT 0
#define CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK 1
#define CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X 5
#define CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y 6
#define CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK 0

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
  X(cuModuleLoadData, "cuModuleLoadData", CUresult,                            \
    (CUmodule *, const void *))                                                \
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
  X(cuMemcpyHtoD, "cuMemcpyHtoD_v2", CUresult,                                 \
    (CUdeviceptr, const void *, size_t))                                       \
  X(cuMemcpyHtoDAsync, "cuMemcpyHtoDAsync_v2", CUresult,                       \
    (CUdeviceptr, const void *, size_t, CUstream))                             \
  X(cuMemcpyDtoH, "cuMemcpyDtoH_v2", CUresult, (void *, CUdeviceptr, size_t))  \
  X(cuStreamCreate, "cuStreamCreate", CUresult, (CUstream *, unsigned))        \
  X(cuStreamDestroy, "cuStreamDestroy_v2", CUresult, (CUstream))               \
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

struct GeozlGpuProgram {
  CUdevice device;
  int deviceIndex;
  CUcontext context;
  CUmodule module;
  CUfunction kernels[GEOZL_GPU_MAX_KERNELS];
  CUdeviceptr params;
  size_t paramsSize;
  size_t kernelCount;
  unsigned maxGridX;
  unsigned maxGridY;
  unsigned maxThreadsPerBlock;
  unsigned maxThreads[GEOZL_GPU_MAX_KERNELS];
  char deviceName[128];
};

struct GeozlGpuBatch {
  GeozlGpuProgram *program;
  GeozlGpuBuffer buffers[GEOZL_GPU_MAX_BUFFERS];
  CUdeviceptr pointers[GEOZL_GPU_MAX_BUFFERS];
  size_t inputOffsets[GEOZL_GPU_MAX_BUFFERS];
  size_t bufferCount;
  void *pinned;
  CUdeviceptr deviceInput;
  size_t inputBytes;
  SlangBuffer params[GEOZL_GPU_MAX_BUFFERS];
  CUstream stream;
  CUevent events[GEOZL_GPU_MAX_KERNELS + 2];
  size_t eventCount;
  int completed;
  int detached;
};

struct GeozlGpuMemory {
  CUdevice device;
  int deviceIndex;
  CUcontext context;
  CUdeviceptr pointer;
  size_t bytes;
  size_t elementBytes;
};

static int push_context(GeozlGpuProgram *program, char *error,
                        size_t errorCapacity) {
  return cuda_ok(cuCtxPushCurrent(program->context), "cuCtxPushCurrent", error,
                 errorCapacity);
}

static int pop_context(char *error, size_t errorCapacity) {
  CUcontext context;
  return cuda_ok(cuCtxPopCurrent(&context), "cuCtxPopCurrent", error,
                 errorCapacity);
}

GeozlGpuProgram *geozl_gpu_program_create(const void *module, size_t moduleSize,
                                          int deviceIndex,
                                          const char *const kernels[],
                                          size_t kernelCount, char *error,
                                          size_t errorCapacity) {
  if (module == NULL || moduleSize == 0 || deviceIndex < 0 || kernels == NULL ||
      kernelCount == 0 || kernelCount > GEOZL_GPU_MAX_KERNELS) {
    geozl_gpu_error(error, errorCapacity, "invalid CUDA program");
    return NULL;
  }
  if (load_driver(error, errorCapacity) != 0 ||
      !cuda_ok(cuInit(0), "cuInit", error, errorCapacity))
    return NULL;

  GeozlGpuProgram *program = calloc(1, sizeof(*program));
  if (program == NULL) {
    geozl_gpu_error(error, errorCapacity, "out of memory");
    return NULL;
  }
  program->deviceIndex = deviceIndex;
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
    geozl_gpu_program_destroy(program);
    return NULL;
  }
  if (maxGridX <= 0 || maxGridY <= 0 || maxThreads <= 0) {
    geozl_gpu_error(error, errorCapacity,
                    "CUDA device reports invalid launch limits");
    geozl_gpu_program_destroy(program);
    return NULL;
  }
  program->maxGridX = (unsigned)maxGridX;
  program->maxGridY = (unsigned)maxGridY;
  program->maxThreadsPerBlock = (unsigned)maxThreads;
  if (!cuda_ok(cuDevicePrimaryCtxRetain(&program->context, program->device),
               "cuDevicePrimaryCtxRetain", error, errorCapacity)) {
    geozl_gpu_program_destroy(program);
    return NULL;
  }
  if (!push_context(program, error, errorCapacity)) {
    geozl_gpu_program_destroy(program);
    return NULL;
  }

  char *ptx = malloc(moduleSize + 1);
  if (ptx == NULL) {
    geozl_gpu_error(error, errorCapacity, "out of memory");
    pop_context(NULL, 0);
    geozl_gpu_program_destroy(program);
    return NULL;
  }
  memcpy(ptx, module, moduleSize);
  ptx[moduleSize] = 0;
  const int loaded = cuda_ok(cuModuleLoadData(&program->module, ptx),
                             "cuModuleLoadData", error, errorCapacity);
  free(ptx);
  if (!loaded) {
    pop_context(NULL, 0);
    geozl_gpu_program_destroy(program);
    return NULL;
  }

  program->kernelCount = kernelCount;
  for (size_t i = 0; i < kernelCount; ++i) {
    int kernelMaxThreads = 0;
    if (kernels[i] == NULL ||
        !cuda_ok(cuModuleGetFunction(&program->kernels[i], program->module,
                                     kernels[i]),
                 "cuModuleGetFunction", error, errorCapacity) ||
        !cuda_ok(cuFuncGetAttribute(&kernelMaxThreads,
                                    CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                    program->kernels[i]),
                 "cuFuncGetAttribute", error, errorCapacity) ||
        kernelMaxThreads <= 0) {
      pop_context(NULL, 0);
      geozl_gpu_program_destroy(program);
      return NULL;
    }
    program->maxThreads[i] = (unsigned)kernelMaxThreads;
  }
  if (!cuda_ok(cuModuleGetGlobal(&program->params, &program->paramsSize,
                                 program->module, "SLANG_globalParams"),
               "cuModuleGetGlobal", error, errorCapacity)) {
    pop_context(NULL, 0);
    geozl_gpu_program_destroy(program);
    return NULL;
  }
  if (!pop_context(error, errorCapacity)) {
    geozl_gpu_program_destroy(program);
    return NULL;
  }
  return program;
}

void geozl_gpu_program_destroy(GeozlGpuProgram *program) {
  if (program == NULL)
    return;
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

const char *geozl_gpu_program_backend(const GeozlGpuProgram *program) {
  return program == NULL ? NULL : "cuda";
}

const char *geozl_gpu_program_device(const GeozlGpuProgram *program) {
  return program == NULL ? NULL : program->deviceName;
}

GeozlGpuBatch *geozl_gpu_batch_create(GeozlGpuProgram *program,
                                      const GeozlGpuBuffer buffers[],
                                      size_t bufferCount, char *error,
                                      size_t errorCapacity) {
  if (program == NULL || !geozl_gpu_buffers_valid(buffers, bufferCount)) {
    geozl_gpu_error(error, errorCapacity, "invalid CUDA buffers");
    return NULL;
  }
  if (!push_context(program, error, errorCapacity))
    return NULL;

  GeozlGpuBatch *batch = calloc(1, sizeof(*batch));
  if (batch == NULL) {
    geozl_gpu_error(error, errorCapacity, "out of memory");
    pop_context(NULL, 0);
    return NULL;
  }
  batch->program = program;
  batch->bufferCount = bufferCount;
  memcpy(batch->buffers, buffers, bufferCount * sizeof(*buffers));
  for (size_t i = 0; i < bufferCount; ++i)
    batch->inputOffsets[i] = SIZE_MAX;

  for (size_t i = 0; i < bufferCount; ++i) {
    if (buffers[i].alias < 0 && (buffers[i].flags & GEOZL_GPU_COPY_IN) != 0) {
      if (buffers[i].bytes > SIZE_MAX - 255) {
        geozl_gpu_error(error, errorCapacity, "CUDA input %zu is too large", i);
        geozl_gpu_batch_destroy(batch);
        pop_context(NULL, 0);
        return NULL;
      }
      const size_t padded = (buffers[i].bytes + 255) & ~(size_t)255;
      if (padded > SIZE_MAX - batch->inputBytes) {
        geozl_gpu_error(error, errorCapacity, "CUDA inputs are too large");
        geozl_gpu_batch_destroy(batch);
        pop_context(NULL, 0);
        return NULL;
      }
      batch->inputOffsets[i] = batch->inputBytes;
      batch->inputBytes += padded;
    }
  }
  if (batch->inputBytes != 0 &&
      (!cuda_ok(cuMemHostAlloc(&batch->pinned, batch->inputBytes, 0),
                "cuMemHostAlloc", error, errorCapacity) ||
       !cuda_ok(cuMemAlloc(&batch->deviceInput, batch->inputBytes),
                "cuMemAlloc", error, errorCapacity))) {
    geozl_gpu_batch_destroy(batch);
    pop_context(NULL, 0);
    return NULL;
  }

  for (size_t i = 0; i < bufferCount; ++i) {
    if (buffers[i].alias >= 0) {
      batch->pointers[i] = batch->pointers[buffers[i].alias];
    } else if (batch->inputOffsets[i] != SIZE_MAX) {
      batch->pointers[i] = batch->deviceInput + batch->inputOffsets[i];
    } else if (!cuda_ok(cuMemAlloc(&batch->pointers[i], buffers[i].bytes),
                        "cuMemAlloc", error, errorCapacity)) {
      geozl_gpu_batch_destroy(batch);
      pop_context(NULL, 0);
      return NULL;
    }
    batch->params[i] = (SlangBuffer){
        batch->pointers[i], buffers[i].bytes / buffers[i].elementBytes};
  }
  if (program->paramsSize != bufferCount * sizeof(SlangBuffer)) {
    geozl_gpu_error(error, errorCapacity,
                    "CUDA module declares %zu bytes of buffers, need %zu",
                    program->paramsSize, bufferCount * sizeof(SlangBuffer));
    geozl_gpu_batch_destroy(batch);
    pop_context(NULL, 0);
    return NULL;
  }
  if (!cuda_ok(cuStreamCreate(&batch->stream, CU_STREAM_NON_BLOCKING),
               "cuStreamCreate", error, errorCapacity)) {
    geozl_gpu_batch_destroy(batch);
    pop_context(NULL, 0);
    return NULL;
  }
  for (size_t i = 0; i < GEOZL_GPU_MAX_KERNELS + 2; ++i) {
    if (!cuda_ok(cuEventCreate(&batch->events[i], CU_EVENT_DEFAULT),
                 "cuEventCreate", error, errorCapacity)) {
      geozl_gpu_batch_destroy(batch);
      pop_context(NULL, 0);
      return NULL;
    }
    ++batch->eventCount;
  }
  if (!pop_context(error, errorCapacity)) {
    geozl_gpu_batch_destroy(batch);
    return NULL;
  }
  return batch;
}

void geozl_gpu_batch_destroy(GeozlGpuBatch *batch) {
  if (batch == NULL)
    return;
  const int pushed = batch->program != NULL &&
                     batch->program->context != NULL &&
                     cuCtxPushCurrent(batch->program->context) == CUDA_SUCCESS;
  if (pushed) {
    for (size_t i = 0; i < batch->eventCount; ++i)
      cuEventDestroy(batch->events[i]);
    if (batch->stream != NULL)
      cuStreamDestroy(batch->stream);
    for (size_t i = 0; i < batch->bufferCount; ++i) {
      if (batch->buffers[i].alias < 0 && batch->inputOffsets[i] == SIZE_MAX &&
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

int geozl_gpu_batch_run(GeozlGpuBatch *batch,
                        const GeozlGpuDispatch dispatches[],
                        size_t dispatchCount, GeozlGpuRunTimes *times,
                        char *error, size_t errorCapacity) {
  if (batch == NULL || batch->detached || times == NULL ||
      !geozl_gpu_dispatches_valid(dispatches, dispatchCount,
                                  batch == NULL ? 0
                                                : batch->program->kernelCount))
    return geozl_gpu_error(error, errorCapacity, "invalid CUDA dispatches");
  for (size_t i = 0; i < dispatchCount; ++i) {
    const GeozlGpuDispatch *dispatch = &dispatches[i];
    if (dispatch->groupsX > batch->program->maxGridX ||
        dispatch->groupsY > batch->program->maxGridY ||
        dispatch->threads > batch->program->maxThreadsPerBlock ||
        dispatch->threads > batch->program->maxThreads[dispatch->kernel])
      return geozl_gpu_error(error, errorCapacity,
                             "CUDA dispatch %zu exceeds device limits", i);
  }
  if (!push_context(batch->program, error, errorCapacity))
    return 1;

  batch->completed = 0;
  int failed = 1;

  if (batch->inputBytes != 0)
    memset(batch->pinned, 0, batch->inputBytes);
  for (size_t i = 0; i < batch->bufferCount; ++i) {
    if (batch->inputOffsets[i] != SIZE_MAX)
      memcpy((uint8_t *)batch->pinned + batch->inputOffsets[i],
             batch->buffers[i].host, batch->buffers[i].bytes);
  }
  if (!cuda_ok(cuMemcpyHtoD(batch->program->params, batch->params,
                            batch->program->paramsSize),
               "cuMemcpyHtoD", error, errorCapacity) ||
      !cuda_ok(cuEventRecord(batch->events[0], batch->stream), "cuEventRecord",
               error, errorCapacity) ||
      (batch->inputBytes != 0 &&
       !cuda_ok(cuMemcpyHtoDAsync(batch->deviceInput, batch->pinned,
                                  batch->inputBytes, batch->stream),
                "cuMemcpyHtoDAsync", error, errorCapacity)) ||
      !cuda_ok(cuEventRecord(batch->events[1], batch->stream), "cuEventRecord",
               error, errorCapacity))
    goto done;

  for (size_t i = 0; i < dispatchCount; ++i) {
    const GeozlGpuDispatch *dispatch = &dispatches[i];
    if (!cuda_ok(cuLaunchKernel(batch->program->kernels[dispatch->kernel],
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

  memset(times, 0, sizeof(*times));
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
  batch->completed = 1;
  failed = 0;

done:
  if (!pop_context(failed ? NULL : error, failed ? 0 : errorCapacity))
    return 1;
  return failed;
}

int geozl_gpu_batch_download(GeozlGpuBatch *batch, char *error,
                             size_t errorCapacity) {
  if (batch == NULL || !batch->completed || batch->detached)
    return geozl_gpu_error(error, errorCapacity, "invalid CUDA batch");
  if (!push_context(batch->program, error, errorCapacity))
    return 1;
  int failed = 0;
  for (size_t i = 0; i < batch->bufferCount; ++i) {
    const GeozlGpuBuffer *buffer = &batch->buffers[i];
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

GeozlGpuMemory *geozl_gpu_batch_take(GeozlGpuBatch *batch, size_t bufferIndex,
                                     char *error, size_t errorCapacity) {
  if (batch == NULL || !batch->completed || batch->detached ||
      bufferIndex >= batch->bufferCount ||
      batch->buffers[bufferIndex].alias >= 0 ||
      (batch->buffers[bufferIndex].flags & GEOZL_GPU_RESULT) == 0 ||
      batch->inputOffsets[bufferIndex] != SIZE_MAX ||
      batch->pointers[bufferIndex] == 0) {
    geozl_gpu_error(error, errorCapacity,
                    "CUDA buffer cannot become an owned result");
    return NULL;
  }

  GeozlGpuMemory *memory = calloc(1, sizeof(*memory));
  if (memory == NULL) {
    geozl_gpu_error(error, errorCapacity, "out of memory");
    return NULL;
  }
  memory->device = batch->program->device;
  memory->deviceIndex = batch->program->deviceIndex;
  if (!cuda_ok(cuDevicePrimaryCtxRetain(&memory->context, memory->device),
               "cuDevicePrimaryCtxRetain", error, errorCapacity)) {
    free(memory);
    return NULL;
  }
  memory->pointer = batch->pointers[bufferIndex];
  memory->bytes = batch->buffers[bufferIndex].bytes;
  memory->elementBytes = batch->buffers[bufferIndex].elementBytes;
  batch->pointers[bufferIndex] = 0;
  batch->detached = 1;
  return memory;
}

void geozl_gpu_memory_destroy(GeozlGpuMemory *memory) {
  if (memory == NULL)
    return;
  if (memory->context != NULL) {
    const int pushed = cuCtxPushCurrent(memory->context) == CUDA_SUCCESS;
    if (pushed) {
      if (memory->pointer != 0)
        cuMemFree(memory->pointer);
      CUcontext context;
      cuCtxPopCurrent(&context);
    }
    cuDevicePrimaryCtxRelease(memory->device);
  }
  free(memory);
}

const char *geozl_gpu_memory_backend(const GeozlGpuMemory *memory) {
  return memory == NULL ? NULL : "cuda";
}

int geozl_gpu_memory_device(const GeozlGpuMemory *memory) {
  return memory == NULL ? -1 : memory->deviceIndex;
}

void *geozl_gpu_memory_data(const GeozlGpuMemory *memory) {
  return memory == NULL ? NULL : (void *)(uintptr_t)memory->pointer;
}

size_t geozl_gpu_memory_capacity(const GeozlGpuMemory *memory) {
  return memory == NULL ? 0 : memory->bytes;
}

size_t geozl_gpu_memory_element_size(const GeozlGpuMemory *memory) {
  return memory == NULL ? 0 : memory->elementBytes;
}

int geozl_gpu_memory_download(const GeozlGpuMemory *memory, void *dst,
                              size_t bytes, char *error, size_t errorCapacity) {
  if (memory == NULL || bytes > memory->bytes || (dst == NULL && bytes != 0))
    return geozl_gpu_error(error, errorCapacity,
                           "invalid CUDA result download");
  if (bytes == 0)
    return 0;
  if (!cuda_ok(cuCtxPushCurrent(memory->context), "cuCtxPushCurrent", error,
               errorCapacity))
    return 1;
  const int failed = !cuda_ok(cuMemcpyDtoH(dst, memory->pointer, bytes),
                              "cuMemcpyDtoH", error, errorCapacity);
  if (!pop_context(failed ? NULL : error, failed ? 0 : errorCapacity))
    return 1;
  return failed;
}
