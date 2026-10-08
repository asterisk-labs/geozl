// CUDA backend of gpu.h through the driver API only: the kernels ship as PTX
// and the driver compiles them for the installed GPU. libcuda is opened at run
// time, so neither the CUDA toolkit nor its headers are needed to build or run.
// Inputs are staged in pinned memory and copied to the device on every run.

#include "gpu.h"

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *gpu_backend_name = "cuda";

double now_seconds(void);

// The few driver API types and entry points this backend uses.
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

#define CU_API(X)                                                              \
  X(cuInit, "cuInit", CUresult, (unsigned))                                    \
  X(cuGetErrorString, "cuGetErrorString", CUresult, (CUresult, const char **)) \
  X(cuDeviceGet, "cuDeviceGet", CUresult, (CUdevice *, int))                   \
  X(cuDeviceGetName, "cuDeviceGetName", CUresult, (char *, int, CUdevice))     \
  X(cuCtxCreate, "cuCtxCreate_v2", CUresult,                                   \
    (CUcontext *, unsigned, CUdevice))                                         \
  X(cuCtxDestroy, "cuCtxDestroy_v2", CUresult, (CUcontext))                    \
  X(cuModuleLoadData, "cuModuleLoadData", CUresult,                            \
    (CUmodule *, const void *))                                                \
  X(cuModuleGetFunction, "cuModuleGetFunction", CUresult,                      \
    (CUfunction *, CUmodule, const char *))                                    \
  X(cuModuleGetGlobal, "cuModuleGetGlobal_v2", CUresult,                       \
    (CUdeviceptr *, size_t *, CUmodule, const char *))                         \
  X(cuMemHostAlloc, "cuMemHostAlloc", CUresult, (void **, size_t, unsigned))   \
  X(cuMemAlloc, "cuMemAlloc_v2", CUresult, (CUdeviceptr *, size_t))            \
  X(cuMemcpyHtoD, "cuMemcpyHtoD_v2", CUresult,                                 \
    (CUdeviceptr, const void *, size_t))                                       \
  X(cuMemcpyHtoDAsync, "cuMemcpyHtoDAsync_v2", CUresult,                       \
    (CUdeviceptr, const void *, size_t, CUstream))                             \
  X(cuMemcpyDtoH, "cuMemcpyDtoH_v2", CUresult, (void *, CUdeviceptr, size_t))  \
  X(cuStreamCreate, "cuStreamCreate", CUresult, (CUstream *, unsigned))        \
  X(cuEventCreate, "cuEventCreate", CUresult, (CUevent *, unsigned))           \
  X(cuEventRecord, "cuEventRecord", CUresult, (CUevent, CUstream))             \
  X(cuEventSynchronize, "cuEventSynchronize", CUresult, (CUevent))             \
  X(cuEventElapsedTime, "cuEventElapsedTime", CUresult,                        \
    (float *, CUevent, CUevent))                                               \
  X(cuLaunchKernel, "cuLaunchKernel", CUresult,                                \
    (CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,   \
     unsigned, CUstream, void **, void **))

#define CU_DECLARE(name, symbol, ret, args) static ret(*name) args;
CU_API(CU_DECLARE)

static int load_driver(void) {
  void *lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  if (!lib) {
    fprintf(stderr, "no NVIDIA driver: %s\n", dlerror());
    return 1;
  }
#define CU_RESOLVE(name, symbol, ret, args)                                    \
  if (!(*(void **)&name = dlsym(lib, symbol))) {                               \
    fprintf(stderr, "libcuda lacks %s\n", symbol);                             \
    return 1;                                                                  \
  }
  CU_API(CU_RESOLVE)
  return 0;
}

#define CK(call)                                                               \
  do {                                                                         \
    CUresult rc_ = (call);                                                     \
    if (rc_ != CUDA_SUCCESS) {                                                 \
      const char *msg_ = NULL;                                                 \
      cuGetErrorString(rc_, &msg_);                                            \
      fprintf(stderr, "%s: %s\n", #call, msg_ ? msg_ : "?");                   \
      return 1;                                                                \
    }                                                                          \
  } while (0)

// Mirrors a buffer in the generated CUDA: a pointer and its element count.
typedef struct {
  CUdeviceptr ptr;
  uint64_t count;
} SlangBuffer;

static char *read_file(const char *path) {
  FILE *fp = fopen(path, "rb");
  if (!fp)
    return NULL;
  if (fseek(fp, 0, SEEK_END) != 0) {
    fclose(fp);
    return NULL;
  }
  const long n = ftell(fp);
  if (n < 0 || fseek(fp, 0, SEEK_SET) != 0) {
    fclose(fp);
    return NULL;
  }
  char *s = malloc((size_t)n + 1);
  if (s == NULL || fread(s, 1, (size_t)n, fp) != (size_t)n) {
    fclose(fp);
    free(s);
    return NULL;
  }
  s[n] = 0;
  fclose(fp);
  return s;
}

static int validate_run(const GpuBuffer *buffers, int nBuffers,
                        const GpuDispatch *dispatches, int nDispatches,
                        int reps, const GpuTimes *times) {
  if (buffers == NULL || dispatches == NULL || times == NULL || reps <= 0 ||
      nBuffers <= 0 || nBuffers > GPU_MAX_BUFFERS || nDispatches <= 0 ||
      nDispatches > GPU_MAX_DISPATCHES) {
    fprintf(stderr, "invalid GPU run dimensions\n");
    return 1;
  }
  for (int i = 0; i < nBuffers; ++i) {
    const GpuBuffer *buffer = &buffers[i];
    if (buffer->bytes == 0 || buffer->elem == 0 ||
        buffer->bytes % buffer->elem != 0 || buffer->aliasOf >= i ||
        buffer->aliasOf < -1) {
      fprintf(stderr, "invalid GPU buffer %d\n", i);
      return 1;
    }
  }
  for (int i = 0; i < nDispatches; ++i) {
    const GpuDispatch *dispatch = &dispatches[i];
    if (dispatch->entry == NULL || dispatch->groupsX == 0 ||
        dispatch->groupsY == 0 || dispatch->threads == 0) {
      fprintf(stderr, "invalid GPU dispatch %d\n", i);
      return 1;
    }
  }
  return 0;
}

int gpu_run(const char *module, GpuBuffer *buffers, int nBuffers,
            const GpuDispatch *dispatches, int nDispatches, int reps,
            GpuTimes *times) {
  if (module == NULL || validate_run(buffers, nBuffers, dispatches, nDispatches,
                                     reps, times) != 0)
    return 1;
  if (load_driver() != 0)
    return 1;
  CK(cuInit(0));
  CUdevice dev;
  CK(cuDeviceGet(&dev, getenv("GPU") ? atoi(getenv("GPU")) : 0));
  CK(cuDeviceGetName(times->device, sizeof(times->device), dev));
  CUcontext ctx;
  CK(cuCtxCreate(&ctx, 0, dev));

  char *ptx = read_file(module);
  if (!ptx) {
    fprintf(stderr, "cannot read %s\n", module);
    return 1;
  }
  CUmodule mod;
  CK(cuModuleLoadData(&mod, ptx));
  CUfunction fns[GPU_MAX_DISPATCHES];
  for (int d = 0; d < nDispatches; ++d)
    CK(cuModuleGetFunction(&fns[d], mod, dispatches[d].entry));
  CUdeviceptr params;
  size_t paramsSize = 0;
  CK(cuModuleGetGlobal(&params, &paramsSize, mod, "SLANG_globalParams"));
  if (paramsSize != (size_t)nBuffers * sizeof(SlangBuffer)) {
    fprintf(stderr, "module declares %zu bytes of buffers, %d given\n",
            paramsSize, nBuffers);
    return 1;
  }

  // Inputs share one pinned staging area and one device copy.
  size_t inBytes = 0;
  size_t inOffset[GPU_MAX_BUFFERS];
  for (int i = 0; i < nBuffers; ++i)
    if (buffers[i].aliasOf < 0 && buffers[i].host && !buffers[i].output) {
      inOffset[i] = inBytes;
      if (buffers[i].bytes > SIZE_MAX - 255) {
        fprintf(stderr, "GPU input buffer %d is too large\n", i);
        return 1;
      }
      const size_t padded = (buffers[i].bytes + 255) / 256 * 256;
      if (padded > SIZE_MAX - inBytes) {
        fprintf(stderr, "GPU input buffers are too large\n");
        return 1;
      }
      inBytes += padded;
    }
  void *pinned = NULL;
  CUdeviceptr dIn = 0;
  if (inBytes) {
    CK(cuMemHostAlloc(&pinned, inBytes, 0));
    CK(cuMemAlloc(&dIn, inBytes));
  }
  SlangBuffer slang[GPU_MAX_BUFFERS];
  CUdeviceptr ptrs[GPU_MAX_BUFFERS];
  for (int i = 0; i < nBuffers; ++i) {
    GpuBuffer *b = &buffers[i];
    if (b->aliasOf >= 0) {
      ptrs[i] = ptrs[b->aliasOf];
    } else if (b->host && !b->output) {
      memcpy((uint8_t *)pinned + inOffset[i], b->host, b->bytes);
      ptrs[i] = dIn + inOffset[i];
    } else {
      CK(cuMemAlloc(&ptrs[i], b->bytes));
    }
    slang[i] = (SlangBuffer){ptrs[i], b->bytes / b->elem};
  }
  CK(cuMemcpyHtoD(params, slang, (size_t)nBuffers * sizeof(SlangBuffer)));

  CUstream stream;
  CK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  CUevent ev[GPU_MAX_DISPATCHES + 2];
  for (int e = 0; e < nDispatches + 2; ++e)
    CK(cuEventCreate(&ev[e], CU_EVENT_DEFAULT));

  times->kernel = times->endToEnd = times->copy = 1e30;
  for (int d = 0; d < nDispatches; ++d)
    times->pass[d] = 1e30;
  for (int r = 0; r < reps; ++r) {
    const double wallStart = now_seconds();
    CK(cuEventRecord(ev[0], stream));
    if (inBytes)
      CK(cuMemcpyHtoDAsync(dIn, pinned, inBytes, stream));
    CK(cuEventRecord(ev[1], stream));
    for (int d = 0; d < nDispatches; ++d) {
      CK(cuLaunchKernel(fns[d], dispatches[d].groupsX, dispatches[d].groupsY, 1,
                        dispatches[d].threads, 1, 1, 0, stream, NULL, NULL));
      CK(cuEventRecord(ev[2 + d], stream));
    }
    CK(cuEventSynchronize(ev[1 + nDispatches]));
    const double wall = now_seconds() - wallStart;
    float ms = 0, total = 0;
    CK(cuEventElapsedTime(&ms, ev[0], ev[1]));
    if (ms / 1e3 < times->copy)
      times->copy = ms / 1e3;
    for (int d = 0; d < nDispatches; ++d) {
      CK(cuEventElapsedTime(&ms, ev[1 + d], ev[2 + d]));
      total += ms;
      if (ms / 1e3 < times->pass[d])
        times->pass[d] = ms / 1e3;
    }
    if (total / 1e3 < times->kernel)
      times->kernel = total / 1e3;
    if (wall < times->endToEnd)
      times->endToEnd = wall;
  }
  for (int i = 0; i < nBuffers; ++i)
    if (buffers[i].output && buffers[i].host && buffers[i].aliasOf < 0)
      CK(cuMemcpyDtoH(buffers[i].host, ptrs[i], buffers[i].bytes));
  CK(cuCtxDestroy(ctx));
  return 0;
}
