#include "gpu/pivco_plan.h"
#include "gpu/runtime.h"

#include "common/endian.h"
#include "openzl/codecs/pivco_huffman/encode_pivco_kernel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail(const char *message) {
  fprintf(stderr, "test_pivco_decode: %s\n", message);
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

static size_t array_bytes(size_t count, size_t elementBytes) {
  return (count == 0 ? 1 : count) * elementBytes;
}

static int run_case(const char *modulePath, size_t blockSize) {
  enum { count = 16384 };
  static const uint8_t weights[] = {2, 1, 1};
  uint8_t *input = malloc(count);
  if (input == NULL)
    return fail("allocation failed");
  for (size_t i = 0; i < count; ++i)
    input[i] = (uint8_t)((i * 17 + i / 19) % 3);

  const int tableLog =
      ZL_PivCoHuffman_computeTableLog(weights, sizeof(weights));
  const size_t payloadCapacity = ZL_PivCoHuffmanEncode_bound(count, blockSize);
  const size_t scratchSize =
      ZL_PivCoHuffmanEncode_scratchElements(count, blockSize);
  uint8_t *payload = calloc(1, payloadCapacity);
  uint8_t *scratch = malloc(scratchSize);
  if (tableLog < 0 || payload == NULL || scratch == NULL) {
    free(scratch);
    free(payload);
    free(input);
    return fail("encoder allocation failed");
  }
  const size_t payloadSize = ZL_PivCoHuffman_encode(
      payload, payloadCapacity, scratch, scratchSize, weights, sizeof(weights),
      tableLog, input, count, blockSize, NULL);
  free(scratch);
  if (payloadSize == SIZE_MAX || payloadCapacity - payloadSize < 64) {
    free(payload);
    free(input);
    return fail("could not encode the test lane");
  }

  CapturedFrame frame = {0};
  frame.headerSize = PLANAR_ZIGZAG_TRANSPOSE_PIVCO_HEADER_FIXED +
                     PLANAR_ZIGZAG_TRANSPOSE_PIVCO_HEADER_LANE;
  geozl_st_le64(frame.header, count);
  frame.header[8] = 1;
  geozl_st_le32(frame.header + 9, count);
  geozl_st_le32(frame.header + 13, 1);
  geozl_st_le32(frame.header + 17, (uint32_t)blockSize);
  frame.header[PLANAR_ZIGZAG_TRANSPOSE_PIVCO_HEADER_FIXED] =
      PLANAR_ZIGZAG_TRANSPOSE_PIVCO_LANE_PIVCO;
  geozl_st_le16(frame.header + PLANAR_ZIGZAG_TRANSPOSE_PIVCO_HEADER_FIXED + 1,
                sizeof(weights));
  geozl_st_le32(frame.header + PLANAR_ZIGZAG_TRANSPOSE_PIVCO_HEADER_FIXED + 3,
                (uint32_t)payloadSize);
  memcpy(frame.weights, weights, sizeof(weights));
  frame.weightsSize = sizeof(weights);
  frame.bitsSize = payloadSize;
  frame.kind = PIVCO_CAPTURE_KIND_PLANAR_ZIGZAG;
  frame.seen = 1;

  const GeozlGpuFrameShape shape = {0, count, count, 1};
  PivcoGpuPlan plan;
  if (pivco_gpu_plan_build(&plan, &frame, &shape, 1, payload, payloadSize, 1)) {
    free(payload);
    free(input);
    return fail("planner rejected the test lane");
  }

  size_t moduleSize;
  void *module = read_file(modulePath, &moduleSize);
  char error[512] = {0};
  static const char *const kernels[] = {"decode_symbols"};
  const char *deviceEnv = getenv("GPU");
  const int device = deviceEnv == NULL ? 0 : atoi(deviceEnv);
  GeozlGpuProgram *program =
      module == NULL
          ? NULL
          : geozl_gpu_program_create(GPU_BACKEND, module, moduleSize, device,
                                     kernels, 1, error, sizeof(error));
  free(module);
  if (program == NULL) {
    pivco_gpu_plan_free(&plan);
    free(payload);
    free(input);
    return fail(module == NULL ? "could not read the shader" : error);
  }

  const size_t laneBytes = plan.laneBytes + 256;
  // Placeholder for unused bindings; the largest one is a GpuJoinTask.
  uint32_t dummy[sizeof(GpuJoinTask) / sizeof(uint32_t)] = {0};
  GeozlGpuBuffer buffers[] = {
      {payload, payloadSize + 64, 1, GEOZL_GPU_COPY_IN, -1},
      {plan.symbols, array_bytes(plan.symbolCount, 1), 1, GEOZL_GPU_COPY_IN,
       -1},
      {plan.trees, array_bytes(plan.treeCount, sizeof(GpuTreeNode)),
       sizeof(GpuTreeNode), GEOZL_GPU_COPY_IN, -1},
      {plan.nodes, array_bytes(plan.nodeCount, sizeof(GpuNodeBits)),
       sizeof(GpuNodeBits), GEOZL_GPU_COPY_IN, -1},
      {plan.pivcoBlocks,
       array_bytes(plan.pivcoBlockCount, sizeof(GpuPivcoBlock)),
       sizeof(GpuPivcoBlock), GEOZL_GPU_COPY_IN, -1},
      {NULL, laneBytes, 1, GEOZL_GPU_RESULT, -1},
      {NULL, laneBytes, 1, 0, 5},
      {dummy, 2 * sizeof(uint32_t), 2 * sizeof(uint32_t), GEOZL_GPU_COPY_IN,
       -1},
      {dummy, sizeof(GpuJoinTask), sizeof(GpuJoinTask), GEOZL_GPU_COPY_IN, -1},
      {dummy, 3 * sizeof(uint32_t), 3 * sizeof(uint32_t), GEOZL_GPU_COPY_IN,
       -1},
      {NULL, laneBytes, 1, 0, -1},
      {NULL, laneBytes, 1, 0, 10},
      {NULL, laneBytes, 1, 0, 10},
      {dummy, 1, 1, GEOZL_GPU_COPY_IN, -1},
      {dummy, 3 * sizeof(uint32_t), 3 * sizeof(uint32_t), GEOZL_GPU_COPY_IN,
       -1},
  };
  GeozlGpuBatch *batch = geozl_gpu_batch_create(
      program, buffers, sizeof(buffers) / sizeof(buffers[0]), error,
      sizeof(error));
  if (batch == NULL) {
    geozl_gpu_program_destroy(program);
    pivco_gpu_plan_free(&plan);
    free(payload);
    free(input);
    return fail(error);
  }

  const GeozlGpuDispatch dispatches[] = {
      {0, (unsigned)plan.pivcoBlockCount, 1, 128},
  };
  GeozlGpuRunTimes times;
  int failed =
      geozl_gpu_batch_run(batch, dispatches, 1, &times, error, sizeof(error));
  GeozlGpuMemory *memory =
      failed ? NULL : geozl_gpu_batch_take(batch, 5, error, sizeof(error));
  geozl_gpu_batch_destroy(batch);
  geozl_gpu_program_destroy(program);

  uint8_t *output = malloc(laneBytes);
  failed = memory == NULL || output == NULL ||
           geozl_gpu_memory_download(memory, output, count, error,
                                     sizeof(error)) != 0 ||
           memcmp(output, input, count) != 0;
  if (failed) {
    if (error[0] == '\0' && memory != NULL && output != NULL) {
      size_t mismatch = 0;
      while (mismatch < count && output[mismatch] == input[mismatch])
        ++mismatch;
      snprintf(error, sizeof(error), "GPU output mismatch at byte %zu",
               mismatch);
    }
    fprintf(stderr, "test_pivco_decode: block %zu: %s\n", blockSize,
            error[0] ? error : "GPU output mismatch");
  }

  geozl_gpu_memory_destroy(memory);
  free(output);
  pivco_gpu_plan_free(&plan);
  free(payload);
  free(input);
  return failed;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: test_pivco_decode pzv1.metal|ptx\n");
    return 2;
  }
  if (run_case(argv[1], 1025) || run_case(argv[1], 1028))
    return 1;
  puts("test_pivco_decode: ok");
  return 0;
}
