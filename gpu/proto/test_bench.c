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

typedef struct {
  uint8_t *base;
  size_t used;
  size_t capacity;
} Bump;

static void *bump_alloc(void *context, size_t bytes) {
  Bump *bump = context;
  const size_t size = (bytes + 15) & ~(size_t)15;
  if (size > bump->capacity - bump->used)
    return NULL;
  void *memory = bump->base + bump->used;
  bump->used += size;
  return memory;
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
        PFOR_GPU_ID};
  }

  PforGpuPlan sequential = {0}, parallel = {0};
  CHECK(pfor_gpu_plan_build(&sequential, frames, frameCount, payload,
                            sizeof(payload), 1) == 0);
  CHECK(pfor_gpu_plan_build_parallel(&parallel, frames, frameCount, payload,
                                     sizeof(payload), 1, frameCount,
                                     NULL) == 0);
  CHECK(sequential.blockCount == parallel.blockCount);
  CHECK(sequential.realBlockCount == parallel.realBlockCount);
  CHECK(sequential.blocksWithExceptions == parallel.blocksWithExceptions);
  CHECK(sequential.exceptionCount == parallel.exceptionCount);
  if (sequential.blockCount == parallel.blockCount)
    CHECK(memcmp(sequential.blocks, parallel.blocks,
                 sequential.blockCount * sizeof(*sequential.blocks)) == 0);

  // The merged blocks land in the caller's memory, which free leaves alone.
  static _Alignas(16) uint8_t arena[4096];
  Bump bump = {arena, 0, sizeof(arena)};
  const GeozlGpuHostAlloc alloc = {bump_alloc, &bump};
  PforGpuPlan staged = {0};
  CHECK(pfor_gpu_plan_build_parallel(&staged, frames, frameCount, payload,
                                     sizeof(payload), 1, 2, &alloc) == 0);
  CHECK(staged.borrowed && (uint8_t *)staged.blocks == arena);
  if (staged.blockCount == sequential.blockCount)
    CHECK(memcmp(sequential.blocks, staged.blocks,
                 staged.blockCount * sizeof(*staged.blocks)) == 0);
  pfor_gpu_plan_free(&staged);
  bump = (Bump){arena, 0, 16};
  CHECK(pfor_gpu_plan_build_parallel(&staged, frames, frameCount, payload,
                                     sizeof(payload), 1, 2, &alloc) != 0);
  CHECK(staged.blocks == NULL && staged.error != NULL);
  pfor_gpu_plan_free(&parallel);
  pfor_gpu_plan_free(&sequential);
}

static void test_band_plan(void) {
  const PforGpuFrame frames[] = {
      {{0, GEOZL_PFOR_BLOCK, GEOZL_PFOR_BLOCK, 1}, 0, 2, PFOR_GPU_ID},
      {{5, 2 * 19 * 2100, 2100, 2}, 64, 4096, PFOR_GPU_PLANAR_ZIGZAG},
  };
  const size_t ids[] = {7, 9};
  PforGpuBandPlan plan;
  CHECK(pfor_gpu_band_plan_build(&plan, frames, 2, ids, NULL) == 0);
  CHECK(plan.frameCount == 4 && plan.stripCount == 20);
  CHECK(plan.blockCount == 2 * (132 + 17 + 9));
  const GpuPforBandFrame *frame = &plan.frames[0];
  CHECK(frame->payloadOffset == 64 && frame->payloadSize == 4096 &&
        frame->blockCount == plan.blockCount && frame->firstBlock == 0 &&
        frame->frame == 9);
  CHECK(plan.frames[1].blockCount == 0 && plan.frames[3].blockCount == 0);
  static const unsigned heights[] = {16, 2, 1};
  static const unsigned blocks[3][3] = {{64, 64, 4}, {8, 8, 1}, {4, 4, 1}};
  static const unsigned rows[] = {0, 16, 18};
  size_t s = 0, next = 0;
  for (unsigned p = 0; p < 2; ++p)
    for (unsigned b = 0; b < 3; ++b)
      for (unsigned x = 0; x < 3; ++x, ++s) {
        const GpuPforStrip *strip = &plan.strips[s];
        CHECK(strip->firstBlock == next && strip->blockCount == blocks[b][x]);
        CHECK(strip->outputBase ==
              5 + p * 19 * 2100 + rows[b] * 2100 + x * 1024);
        CHECK(strip->width == 2100 && strip->height == heights[b] &&
              strip->frame == 9 && strip->columns == (x < 2 ? 1024 : 52));
        next += strip->blockCount;
      }
  CHECK(plan.strips[18].blockCount == 0 && plan.strips[19].blockCount == 0);
  CHECK(plan.chunkCount == 4 && plan.chunks[0].frame == 0 &&
        plan.chunks[0].first == 0 && plan.chunks[0].count == 256 &&
        plan.chunks[1].first == 256 && plan.chunks[1].count == 60 &&
        plan.chunks[1].firstChunk == 0 && plan.chunks[2].count == 0);
  pfor_gpu_band_plan_free(&plan);

  PforGpuPlan blockPlan;
  const uint8_t payload[4096 + 64] = {0};
  CHECK(pfor_gpu_plan_build(&blockPlan, &frames[1], 1, payload, 4096 + 64, 2) !=
        0);
  PforGpuFrame bad = frames[1];
  bad.shape.width = 0;
  CHECK(pfor_gpu_band_plan_build(&plan, &bad, 1, NULL, NULL) != 0 &&
        plan.errorFrame == 0 && plan.strips == NULL);
  bad = frames[1];
  bad.shape.width = 2099;
  CHECK(pfor_gpu_band_plan_build(&plan, &bad, 1, NULL, NULL) != 0);
  bad = frames[1];
  bad.payloadSize = 3 * 2 * (132 + 17 + 9);
  CHECK(pfor_gpu_band_plan_build(&plan, &bad, 1, NULL, NULL) == 0);
  pfor_gpu_band_plan_free(&plan);
  bad.payloadSize -= 1;
  CHECK(pfor_gpu_band_plan_build(&plan, &bad, 1, NULL, NULL) != 0);
}

int main(void) {
  test_pfor_block_info();
  test_parallel_plan();
  test_band_plan();
  if (failures != 0)
    return 1;
  printf("test_bench: ok\n");
  return 0;
}
