#include "gpu/runtime.h"
#include "gpu/staging_arena.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int fail(const char *message) {
  fprintf(stderr, "test_runtime_metal: %s\n", message);
  return 1;
}

// Inputs in staging are bound in place at their offset, and a buffer that
// does not lie aligned inside its staging is refused.
static int check_staging(void) {
  static const char source[] =
      "#include <metal_stdlib>\n"
      "using namespace metal;\n"
      "kernel void twice(device const uint *in [[buffer(0)]], "
      "device uint *out [[buffer(1)]], "
      "uint i [[thread_position_in_grid]]) { out[i] = in[i] * 2; }\n";
  const char *kernels[] = {"twice"};
  char error[256] = {0};
  GeozlGpuProgram *program =
      geozl_gpu_program_create(GEOZL_GPU_BACKEND_METAL, source, strlen(source),
                               0, kernels, 1, error, sizeof(error));
  if (program == NULL)
    return fail(error);
  GeozlGpuStaging *staging =
      geozl_gpu_staging_create(program, 4096, error, sizeof(error));
  if (staging == NULL || geozl_gpu_staging_capacity(staging) != 4096) {
    geozl_gpu_program_destroy(program);
    return fail("staging was not created");
  }

  enum { COUNT = 64 };
  uint8_t *data = geozl_gpu_staging_data(staging);
  uint32_t *input = (uint32_t *)(data + GEOZL_GPU_STAGING_ALIGNMENT);
  for (uint32_t i = 0; i < COUNT; ++i)
    input[i] = i + 7;
  uint32_t output[COUNT] = {0};
  GeozlGpuBuffer buffers[] = {
      {input, sizeof(output), sizeof(uint32_t), GEOZL_GPU_COPY_IN, -1, staging},
      {output, sizeof(output), sizeof(uint32_t), GEOZL_GPU_COPY_OUT, -1, NULL},
  };
  const GeozlGpuBuffer refused[][2] = {
      {{data + 4, sizeof(output), 4, GEOZL_GPU_COPY_IN, -1, staging},
       buffers[1]},
      {{data + 4096 - GEOZL_GPU_STAGING_ALIGNMENT, 2 * sizeof(output), 4,
        GEOZL_GPU_COPY_IN, -1, staging},
       buffers[1]},
      {buffers[0], {data, sizeof(output), 4, GEOZL_GPU_COPY_OUT, -1, staging}},
  };
  for (size_t i = 0; i < sizeof(refused) / sizeof(refused[0]); ++i) {
    GeozlGpuBatch *bad =
        geozl_gpu_batch_create(program, refused[i], 2, error, sizeof(error));
    if (bad != NULL || strstr(error, "outside staging") == NULL) {
      geozl_gpu_batch_destroy(bad);
      geozl_gpu_staging_destroy(staging);
      geozl_gpu_program_destroy(program);
      return fail("a buffer outside staging was accepted");
    }
  }

  GeozlGpuBatch *batch =
      geozl_gpu_batch_create(program, buffers, 2, error, sizeof(error));
  const GeozlGpuDispatch dispatch = {0, 1, 1, COUNT};
  GeozlGpuRunTimes times;
  const int failed =
      batch == NULL ||
      geozl_gpu_batch_run(batch, &dispatch, 1, &times, error, sizeof(error)) ||
      geozl_gpu_batch_download(batch, error, sizeof(error));
  geozl_gpu_batch_destroy(batch);
  geozl_gpu_staging_destroy(staging);
  geozl_gpu_program_destroy(program);
  if (failed)
    return fail(error);
  for (uint32_t i = 0; i < COUNT; ++i) {
    if (output[i] != (i + 7) * 2)
      return fail("a staged input was not read in place");
  }
  return 0;
}

// Allocations that overflow a chunk open another, and reset merges them into
// one chunk that holds the same allocations again.
static int check_arena(void) {
  static const char source[] =
      "kernel void idle(device uint *out [[buffer(0)]]) {}\n";
  const char *kernels[] = {"idle"};
  char error[256] = {0};
  GeozlGpuProgram *program =
      geozl_gpu_program_create(GEOZL_GPU_BACKEND_METAL, source, strlen(source),
                               0, kernels, 1, error, sizeof(error));
  if (program == NULL)
    return fail(error);
  GeozlGpuArena arena;
  geozl_gpu_arena_init(&arena, program);
  enum { PART = 600 << 10 };
  int failed = 0;
  for (int round = 0; round < 2 && !failed; ++round) {
    geozl_gpu_arena_reset(&arena);
    for (int i = 0; i < 3 && !failed; ++i) {
      uint8_t *memory = geozl_gpu_arena_alloc(&arena, PART + 1);
      GeozlGpuStaging *staging = geozl_gpu_arena_staging(&arena, memory);
      failed = memory == NULL || staging == NULL ||
               (uintptr_t)memory % GEOZL_GPU_STAGING_ALIGNMENT != 0 ||
               geozl_gpu_arena_staging(&arena, memory + PART) != staging;
    }
    failed |= arena.count != (round == 0 ? 3u : 1u);
  }
  failed |= geozl_gpu_arena_alloc(&arena, 0) != NULL ||
            strcmp(geozl_gpu_arena_error(&arena), "invalid GPU staging size");
  geozl_gpu_arena_destroy(&arena);
  geozl_gpu_program_destroy(program);
  return failed ? fail("the staging arena did not merge its chunks") : 0;
}

int main(void) {
  if (check_staging() != 0 || check_arena() != 0)
    return 1;
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
