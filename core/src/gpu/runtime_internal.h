#ifndef GEOZL_GPU_RUNTIME_INTERNAL_H
#define GEOZL_GPU_RUNTIME_INTERNAL_H

#include "runtime.h"

#include <stdarg.h>
#include <stdio.h>

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
