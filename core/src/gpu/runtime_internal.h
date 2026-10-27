#ifndef GEOZL_GPU_RUNTIME_INTERNAL_H
#define GEOZL_GPU_RUNTIME_INTERNAL_H

#include "runtime.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

// Backends extend these headers by embedding them as their first member.
// runtime.c validates every call and keeps the shared state before it reaches
// a backend, so a backend checks only what is specific to its device.
typedef struct GeozlGpuBackendOps GeozlGpuBackendOps;

struct GeozlGpuProgram {
  const GeozlGpuBackendOps *ops;
  int device;
  size_t kernelCount;
};

struct GeozlGpuBatch {
  GeozlGpuProgram *program;
  GeozlGpuBuffer buffers[GEOZL_GPU_MAX_BUFFERS];
  size_t bufferCount;
  int completed;
  int detached;
};

struct GeozlGpuMemory {
  const GeozlGpuBackendOps *ops;
  int device;
  size_t bytes;
  size_t elementBytes;
};

struct GeozlGpuStaging {
  GeozlGpuProgram *program;
  void *data;
  size_t bytes;
};

struct GeozlGpuBackendOps {
  const char *name;
  GeozlGpuProgram *(*program_create)(const void *module, size_t moduleSize,
                                     int device, const char *const kernels[],
                                     size_t kernelCount, char *error,
                                     size_t errorCapacity);
  void (*program_destroy)(GeozlGpuProgram *program);
  const char *(*program_device)(const GeozlGpuProgram *program);
  // Returns staging with data set; runtime.c fills the other fields.
  GeozlGpuStaging *(*staging_create)(GeozlGpuProgram *program, size_t bytes,
                                     char *error, size_t errorCapacity);
  void (*staging_destroy)(GeozlGpuStaging *staging);
  GeozlGpuBatch *(*batch_create)(GeozlGpuProgram *program,
                                 const GeozlGpuBuffer buffers[],
                                 size_t bufferCount, char *error,
                                 size_t errorCapacity);
  void (*batch_destroy)(GeozlGpuBatch *batch);
  // Checks device limits before batch_run, so a rejected call leaves the
  // previous result in place.
  int (*dispatches_fit)(const GeozlGpuProgram *program,
                        const GeozlGpuDispatch dispatches[],
                        size_t dispatchCount, char *error,
                        size_t errorCapacity);
  int (*batch_run)(GeozlGpuBatch *batch, const GeozlGpuDispatch dispatches[],
                   size_t dispatchCount, GeozlGpuRunTimes *times, char *error,
                   size_t errorCapacity);
  int (*batch_download)(GeozlGpuBatch *batch, char *error,
                        size_t errorCapacity);
  // Returns the backend memory with only its own fields set.
  GeozlGpuMemory *(*batch_take)(GeozlGpuBatch *batch, size_t buffer,
                                char *error, size_t errorCapacity);
  void (*memory_destroy)(GeozlGpuMemory *memory);
  void *(*memory_data)(const GeozlGpuMemory *memory);
  int (*memory_download)(const GeozlGpuMemory *memory, void *dst, size_t bytes,
                         char *error, size_t errorCapacity);
};

#ifdef GEOZL_GPU_METAL
extern const GeozlGpuBackendOps geozl_gpu_metal_ops;
#endif
#ifdef GEOZL_GPU_CUDA
extern const GeozlGpuBackendOps geozl_gpu_cuda_ops;
#endif

// Backends call this first so their destroy path sees the buffer list.
static inline void geozl_gpu_batch_init(GeozlGpuBatch *batch,
                                        GeozlGpuProgram *program,
                                        const GeozlGpuBuffer buffers[],
                                        size_t bufferCount) {
  batch->program = program;
  memcpy(batch->buffers, buffers, bufferCount * sizeof(*buffers));
  batch->bufferCount = bufferCount;
}

static inline int geozl_gpu_error(char *error, size_t capacity,
                                  const char *format, ...) {
  if (error != NULL && capacity != 0) {
    va_list args;
    va_start(args, format);
    vsnprintf(error, capacity, format, args);
    va_end(args);
  }
  return 1;
}

static inline int geozl_gpu_buffers_valid(const GeozlGpuBuffer buffers[],
                                          size_t count) {
  if (buffers == NULL || count == 0 || count > GEOZL_GPU_MAX_BUFFERS)
    return 0;
  for (size_t i = 0; i < count; ++i) {
    const GeozlGpuBuffer *buffer = &buffers[i];
    if (buffer->bytes == 0 || buffer->elementBytes == 0 ||
        buffer->bytes % buffer->elementBytes != 0 || buffer->alias < -1 ||
        buffer->alias >= (int)i ||
        (buffer->flags &
         ~(GEOZL_GPU_COPY_IN | GEOZL_GPU_COPY_OUT | GEOZL_GPU_RESULT)) != 0 ||
        ((buffer->flags & (GEOZL_GPU_COPY_IN | GEOZL_GPU_COPY_OUT)) != 0 &&
         buffer->host == NULL) ||
        (buffer->host != NULL &&
         (buffer->flags & (GEOZL_GPU_COPY_IN | GEOZL_GPU_COPY_OUT)) == 0) ||
        (buffer->alias >= 0 && (buffer->host != NULL || buffer->flags != 0 ||
                                buffer->bytes > buffers[buffer->alias].bytes)))
      return 0;
  }
  return 1;
}

static inline int
geozl_gpu_dispatches_valid(const GeozlGpuDispatch dispatches[], size_t count,
                           size_t kernelCount) {
  if (dispatches == NULL || count == 0 || count > GEOZL_GPU_MAX_KERNELS)
    return 0;
  for (size_t i = 0; i < count; ++i) {
    const GeozlGpuDispatch *dispatch = &dispatches[i];
    if (dispatch->kernel >= kernelCount || dispatch->groupsX == 0 ||
        dispatch->groupsY == 0 || dispatch->threads == 0)
      return 0;
  }
  return 1;
}

#endif // GEOZL_GPU_RUNTIME_INTERNAL_H
