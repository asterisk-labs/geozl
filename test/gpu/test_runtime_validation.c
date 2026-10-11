#include "gpu/runtime_internal.h"

#include <stdint.h>
#include <stdio.h>

static int failures;

#define CHECK(c)                                                               \
  do {                                                                         \
    if (!(c)) {                                                                \
      fprintf(stderr, "test_runtime_validation:%d: %s\n", __LINE__, #c);       \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

int main(void) {
  uint32_t input[4] = {0};
  GeozlGpuBuffer buffers[] = {
      {input, sizeof(input), sizeof(input[0]), GEOZL_GPU_COPY_IN, -1},
      {NULL, sizeof(input), 1, 0, 0},
  };
  CHECK(geozl_gpu_buffers_valid(buffers, 2));

  buffers[1].bytes = sizeof(input) + 1;
  CHECK(!geozl_gpu_buffers_valid(buffers, 2));
  buffers[1].bytes = sizeof(input);
  buffers[1].flags = GEOZL_GPU_COPY_OUT;
  CHECK(!geozl_gpu_buffers_valid(buffers, 2));

  const GeozlGpuDispatch dispatch = {0, 1, 1, 128};
  CHECK(geozl_gpu_dispatches_valid(&dispatch, 1, 1));
  const GeozlGpuDispatch badKernel = {1, 1, 1, 128};
  CHECK(!geozl_gpu_dispatches_valid(&badKernel, 1, 1));
  const GeozlGpuDispatch noGroups = {0, 0, 1, 128};
  CHECK(!geozl_gpu_dispatches_valid(&noGroups, 1, 1));

  if (failures != 0)
    return 1;
  puts("test_runtime_validation: ok");
  return 0;
}
