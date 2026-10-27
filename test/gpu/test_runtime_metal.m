#include "gpu/runtime.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int fail(const char *message) {
  fprintf(stderr, "test_runtime_metal: %s\n", message);
  return 1;
}

int main(void) {
  static const char source[] =
      "#include <metal_stdlib>\n"
      "using namespace metal;\n"
      "kernel void fill(device uint *out [[buffer(0)]], "
      "uint i [[thread_position_in_grid]]) { out[i] = i * 3 + 1; }\n";
  const char *kernels[] = {"fill"};
  char error[256] = {0};
  if (!geozl_gpu_backend_available(GEOZL_GPU_BACKEND_METAL) ||
      geozl_gpu_backend_available(GEOZL_GPU_BACKEND_CUDA) ||
      geozl_gpu_program_create(GEOZL_GPU_BACKEND_CUDA, source, strlen(source),
                               0, kernels, 1, error, sizeof(error)) != NULL ||
      strcmp(error, "GPU backend is not built in") != 0)
    return fail("a backend that is not built in was accepted");
  GeozlGpuProgram *program =
      geozl_gpu_program_create(GEOZL_GPU_BACKEND_METAL, source, strlen(source),
                               0, kernels, 1, error, sizeof(error));
  if (program == NULL)
    return fail(error);

  enum { COUNT = 64 };
  const GeozlGpuBuffer buffers[] = {
      {NULL, COUNT * sizeof(uint32_t), sizeof(uint32_t), GEOZL_GPU_RESULT, -1},
  };
  GeozlGpuBatch *batch =
      geozl_gpu_batch_create(program, buffers, 1, error, sizeof(error));
  if (batch == NULL) {
    geozl_gpu_program_destroy(program);
    return fail(error);
  }

  GeozlGpuMemory *early = geozl_gpu_batch_take(batch, 0, error, sizeof(error));
  if (early != NULL) {
    geozl_gpu_memory_destroy(early);
    geozl_gpu_batch_destroy(batch);
    geozl_gpu_program_destroy(program);
    return fail("accepted a result before the batch ran");
  }

  const GeozlGpuDispatch dispatch = {0, 1, 1, COUNT};
  GeozlGpuRunTimes times;
  if (geozl_gpu_batch_run(batch, &dispatch, 1, &times, error, sizeof(error)) !=
      0) {
    geozl_gpu_batch_destroy(batch);
    geozl_gpu_program_destroy(program);
    return fail(error);
  }

  GeozlGpuMemory *memory = geozl_gpu_batch_take(batch, 0, error, sizeof(error));
  if (memory == NULL) {
    geozl_gpu_batch_destroy(batch);
    geozl_gpu_program_destroy(program);
    return fail(error);
  }
  if (strcmp(geozl_gpu_memory_backend(memory), "metal") != 0 ||
      geozl_gpu_memory_device(memory) != 0 ||
      geozl_gpu_memory_data(memory) == NULL ||
      geozl_gpu_memory_capacity(memory) != sizeof(uint32_t) * COUNT ||
      geozl_gpu_memory_element_size(memory) != sizeof(uint32_t)) {
    geozl_gpu_memory_destroy(memory);
    geozl_gpu_batch_destroy(batch);
    geozl_gpu_program_destroy(program);
    return fail("wrong result metadata");
  }
  if (geozl_gpu_batch_run(batch, &dispatch, 1, &times, error, sizeof(error)) ==
      0) {
    geozl_gpu_memory_destroy(memory);
    geozl_gpu_batch_destroy(batch);
    geozl_gpu_program_destroy(program);
    return fail("accepted a detached batch");
  }

  // The result owns its device allocation. Neither the batch nor the program
  // has to stay alive while a framework consumes it.
  geozl_gpu_batch_destroy(batch);
  geozl_gpu_program_destroy(program);

  uint32_t output[COUNT];
  if (geozl_gpu_memory_download(memory, output, sizeof(output), error,
                                sizeof(error)) != 0) {
    geozl_gpu_memory_destroy(memory);
    return fail(error);
  }
  for (uint32_t i = 0; i < COUNT; ++i) {
    if (output[i] != i * 3 + 1) {
      geozl_gpu_memory_destroy(memory);
      return fail("downloaded values differ");
    }
  }

  geozl_gpu_memory_destroy(memory);
  puts("test_runtime_metal: ok");
  return 0;
}
