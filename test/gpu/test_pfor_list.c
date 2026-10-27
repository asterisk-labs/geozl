#include "gpu/pfor_plan.h"
#include "gpu/runtime.h"

#include "pfor/pfor_check.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail(const char *message) {
  fprintf(stderr, "test_pfor_list: %s\n", message);
  return 1;
}

static void *read_file(const char *path, size_t *size) {
  FILE *file = fopen(path, "rb");
  if (file == NULL || fseek(file, 0, SEEK_END) != 0)
    return NULL;
  const long length = ftell(file);
  if (length <= 0 || fseek(file, 0, SEEK_SET) != 0) {
    fclose(file);
    return NULL;
  }
  void *data = malloc((size_t)length);
  if (data == NULL || fread(data, 1, (size_t)length, file) != (size_t)length) {
    free(data);
    fclose(file);
    return NULL;
  }
  fclose(file);
  *size = (size_t)length;
  return data;
}

static void put_bits(uint8_t *dst, size_t *bit, unsigned width,
                     unsigned value) {
  for (unsigned i = 0; i < width; ++i, ++*bit)
    dst[*bit >> 3] |= (uint8_t)(((value >> i) & 1u) << (*bit & 7));
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: test_pfor_list pzp1.metal|ptx\n");
    return 2;
  }

  enum { exceptions = 81, exceptionBits = 7 };
  const size_t highBytes = (exceptions * exceptionBits + 7) / 8;
  const size_t blockSize =
      3 + exceptions + highBytes + geozl_pfor_body_bytes(1);
  uint8_t *payload = calloc(1, blockSize + 64);
  if (payload == NULL)
    return fail("allocation failed");
  payload[0] = 1;
  payload[1] = exceptions;
  payload[2] = 0x80u | exceptionBits;
  for (unsigned i = 0; i < exceptions; ++i)
    payload[3 + i] = (uint8_t)i;
  size_t bit = 0;
  uint8_t *high = payload + 3 + exceptions;
  for (unsigned i = 0; i < exceptions; ++i)
    put_bits(high, &bit, exceptionBits, 1);

  const PforGpuFrame frame = {{0, GEOZL_PFOR_BLOCK, GEOZL_PFOR_BLOCK, 1},
                              0,
                              blockSize,
                              PFOR_GPU_PLANAR_ZIGZAG};
  PforGpuPlan plan;
  if (pfor_gpu_plan_build(&plan, &frame, 1, payload, blockSize, 1)) {
    free(payload);
    return fail("planner rejected a valid 81-exception list");
  }

  size_t moduleSize;
  void *module = read_file(argv[1], &moduleSize);
  char error[512] = {0};
  static const char *const kernels[] = {"unpack_blocks"};
  const char *deviceEnv = getenv("GPU");
  const int device = deviceEnv == NULL ? 0 : atoi(deviceEnv);
  GeozlGpuProgram *program =
      module == NULL
          ? NULL
          : geozl_gpu_program_create(GPU_BACKEND, module, moduleSize, device,
                                     kernels, 1, error, sizeof(error));
  free(module);
  if (program == NULL) {
    pfor_gpu_plan_free(&plan);
    free(payload);
    return fail(module == NULL ? "could not read the shader" : error);
  }

  uint32_t dummyPlane[3] = {0};
  uint32_t dummyRaw[3] = {0};
  GeozlGpuBuffer buffers[] = {
      {payload, blockSize + 64, 1, GEOZL_GPU_COPY_IN, -1},
      {plan.blocks, plan.blockCount * sizeof(*plan.blocks),
       sizeof(*plan.blocks), GEOZL_GPU_COPY_IN, -1},
      {dummyPlane, sizeof(dummyPlane), sizeof(dummyPlane), GEOZL_GPU_COPY_IN,
       -1},
      {NULL, GEOZL_PFOR_BLOCK, 1, GEOZL_GPU_RESULT, -1},
      {NULL, GEOZL_PFOR_BLOCK, 1, 0, 3},
      {dummyRaw, 1, 1, GEOZL_GPU_COPY_IN, -1},
      {dummyRaw, sizeof(dummyRaw), sizeof(dummyRaw), GEOZL_GPU_COPY_IN, -1},
  };
  GeozlGpuBatch *batch = geozl_gpu_batch_create(
      program, buffers, sizeof(buffers) / sizeof(buffers[0]), error,
      sizeof(error));
  const GeozlGpuDispatch dispatch = {0, (unsigned)(plan.blockCount / 4), 1,
                                     128};
  GeozlGpuRunTimes times;
  int failed = batch == NULL || geozl_gpu_batch_run(batch, &dispatch, 1, &times,
                                                    error, sizeof(error)) != 0;
  // A dispatch the device rejects must leave the finished result takeable.
  if (!failed) {
    const GeozlGpuDispatch tooWide = {0, 1, 1, 4096};
    if (geozl_gpu_batch_run(batch, &tooWide, 1, &times, error, sizeof(error)) ==
        0) {
      snprintf(error, sizeof(error), "a 4096-thread group was accepted");
      failed = 1;
    } else {
      error[0] = '\0';
    }
  }
  if (!failed && strcmp(geozl_gpu_program_backend(program), "cuda") == 0) {
    const GeozlGpuDispatch tooTall = {0, 1, 65536, 128};
    if (geozl_gpu_batch_run(batch, &tooTall, 1, &times, error, sizeof(error)) ==
            0 ||
        strstr(error, "exceeds device limits") == NULL) {
      snprintf(error, sizeof(error), "CUDA accepted a grid taller than 65535");
      failed = 1;
    } else {
      error[0] = '\0';
    }
  }
  GeozlGpuMemory *memory =
      failed ? NULL : geozl_gpu_batch_take(batch, 3, error, sizeof(error));
  geozl_gpu_batch_destroy(batch);
  geozl_gpu_program_destroy(program);

  uint8_t output[GEOZL_PFOR_BLOCK];
  failed = memory == NULL ||
           geozl_gpu_memory_download(memory, output, sizeof(output), error,
                                     sizeof(error)) != 0;
  for (size_t i = 0; i < sizeof(output) && !failed; ++i) {
    const uint8_t expected = i < exceptions ? 1 : 0;
    if (output[i] != expected) {
      snprintf(error, sizeof(error), "GPU output mismatch at byte %zu", i);
      failed = 1;
    }
  }
  if (failed)
    fprintf(stderr, "test_pfor_list: %s\n",
            error[0] ? error : "GPU decode failed");

  geozl_gpu_memory_destroy(memory);
  pfor_gpu_plan_free(&plan);
  free(payload);
  if (failed)
    return 1;
  puts("test_pfor_list: ok");
  return 0;
}
