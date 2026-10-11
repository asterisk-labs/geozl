#include "bench.h"
#include "gpu/pfor_plan.h"
#include "pfor/pfor_check.h"

#include <stdio.h>
#include <string.h>

static int failures;
const char *gpu_backend_name = "test";

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);     \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

static void test_pfor_block_info(void) {
  uint8_t block[258];
  PforGpuBlockInfo info;

  memset(block, 0, sizeof(block));
  CHECK(pfor_gpu_block_info(block, 2, 1, &info) == 0);
  CHECK(info.bytes == 2 && info.exceptions == 0);

  block[0] = 8;
  CHECK(pfor_gpu_block_info(block, sizeof(block), 1, &info) == 0);
  CHECK(info.bytes == sizeof(block));
  CHECK(pfor_gpu_block_info(block, sizeof(block) - 1, 1, &info) != 0);

  block[0] = 9;
  CHECK(pfor_gpu_block_info(block, sizeof(block), 1, &info) != 0);

  memset(block, 0, sizeof(block));
  block[0] = 1;
  block[1] = 1;
  block[2] = 7; // bitmap mode, seven high bits
  block[3] = 1;
  CHECK(pfor_gpu_block_info(block, 68, 1, &info) == 0);
  CHECK(info.bytes == 68 && info.exceptions == 1);
  CHECK(pfor_gpu_block_info(block, 67, 1, &info) != 0);

  block[3] = 3; // two set positions but nexc is one
  CHECK(pfor_gpu_block_info(block, 68, 1, &info) != 0);
  block[3] = 1;

  block[2] = 0; // an exception must carry at least one high bit
  CHECK(pfor_gpu_block_info(block, sizeof(block), 1, &info) != 0);
  block[2] = 8; // low and high widths exceed uint8
  CHECK(pfor_gpu_block_info(block, sizeof(block), 1, &info) != 0);

  block[2] = 0x87; // list mode, one position byte and one data byte
  CHECK(pfor_gpu_block_info(block, 37, 1, &info) == 0);
  CHECK(info.bytes == 37 && info.exceptions == 1);

  memset(block, 0, sizeof(block));
  block[0] = 1;
  block[1] = 3;
  block[2] = 0x87;
  block[3] = 7;
  block[4] = 7;
  block[5] = 9;
  CHECK(pfor_gpu_block_info(block, sizeof(block), 1, &info) != 0);
  block[4] = 8;
  CHECK(pfor_gpu_block_info(block, sizeof(block), 1, &info) == 0);
}

static void test_parallel_plan(void) {
  enum { frameCount = 4 };
  const uint8_t payload[] = {0, 0};
  PforGpuFrame frames[frameCount];
  for (size_t i = 0; i < frameCount; ++i) {
    frames[i] = (PforGpuFrame){
        {i * GEOZL_PFOR_BLOCK, GEOZL_PFOR_BLOCK, GEOZL_PFOR_BLOCK, 1},
        0,
        sizeof(payload),
        PFOR_GPU_PLANAR_ZIGZAG};
  }

  PforGpuPlan sequential = {0}, parallel = {0};
  CHECK(pfor_gpu_plan_build(&sequential, frames, frameCount, payload,
                            sizeof(payload), 1) == 0);
  CHECK(pfor_gpu_plan_build_parallel(&parallel, frames, frameCount, payload,
                                     sizeof(payload), 1, frameCount) == 0);
  CHECK(sequential.blockCount == parallel.blockCount);
  CHECK(sequential.realBlockCount == parallel.realBlockCount);
  CHECK(sequential.blocksWithExceptions == parallel.blocksWithExceptions);
  CHECK(sequential.exceptionCount == parallel.exceptionCount);
  if (sequential.blockCount == parallel.blockCount)
    CHECK(memcmp(sequential.blocks, parallel.blocks,
                 sequential.blockCount * sizeof(*sequential.blocks)) == 0);
  pfor_gpu_plan_free(&parallel);
  pfor_gpu_plan_free(&sequential);
}

int main(void) {
  test_pfor_block_info();
  test_parallel_plan();
  if (failures != 0)
    return 1;
  printf("test_bench: ok\n");
  return 0;
}
