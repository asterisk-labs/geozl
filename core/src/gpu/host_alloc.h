#ifndef GEOZL_GPU_HOST_ALLOC_H
#define GEOZL_GPU_HOST_ALLOC_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Where planners put the arrays the kernels read. GPU decoders pass staging, so
// a plan uploads without a copy; that memory stays theirs. NULL means calloc,
// and the plan owns its arrays.
typedef struct {
  void *(*alloc)(void *context, size_t bytes);
  void *context;
} GeozlGpuHostAlloc;

// Zeroed storage for count elements, or NULL.
static inline void *geozl_gpu_host_calloc(const GeozlGpuHostAlloc *alloc,
                                          size_t count, size_t elementBytes) {
  if (count == 0 || elementBytes == 0 || count > SIZE_MAX / elementBytes)
    return NULL;
  if (alloc == NULL)
    return calloc(count, elementBytes);
  void *memory = alloc->alloc(alloc->context, count * elementBytes);
  if (memory != NULL)
    memset(memory, 0, count * elementBytes);
  return memory;
}

#endif // GEOZL_GPU_HOST_ALLOC_H
