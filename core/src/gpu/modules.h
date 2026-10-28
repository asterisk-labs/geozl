#ifndef GEOZL_GPU_MODULES_H
#define GEOZL_GPU_MODULES_H

#include <stddef.h>

// Kernels compiled by tools/build_gpu_modules.py, PTX on CUDA and Metal source
// on Metal, named after their program and element width, such as pfor2.
typedef struct {
  const char *name;
  const unsigned char *data;
  size_t size;
} GeozlGpuModule;

extern const GeozlGpuModule geozl_gpu_modules[];
extern const size_t geozl_gpu_module_count;

#endif // GEOZL_GPU_MODULES_H
