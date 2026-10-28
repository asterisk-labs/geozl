#include "gpu/pivco_plan.h"

#include "common/endian.h"
#include "openzl/codecs/pivco_huffman/encode_pivco_kernel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail(const char *message) {
  fprintf(stderr, "test_pivco_plan: %s\n", message);
  return 1;
}

static int make_lane(uint8_t **payload, size_t *payloadSize, uint8_t **scratch,
                     const uint8_t *input, size_t count, size_t blockSize,
                     const uint8_t *weights, size_t weightsSize) {
  const int tableLog = ZL_PivCoHuffman_computeTableLog(weights, weightsSize);
  const size_t capacity = ZL_PivCoHuffmanEncode_bound(count, blockSize);
  const size_t scratchSize =
      ZL_PivCoHuffmanEncode_scratchElements(count, blockSize);
  if (tableLog < 0 || capacity == SIZE_MAX || scratchSize == SIZE_MAX)
    return 1;

  *payload = malloc(capacity);
  *scratch = malloc(scratchSize);
  if (*payload == NULL || *scratch == NULL)
    return 1;
  *payloadSize = ZL_PivCoHuffman_encode(
      *payload, capacity, *scratch, scratchSize, weights, weightsSize, tableLog,
      input, count, blockSize, NULL);
  return *payloadSize == SIZE_MAX;
}

static void make_frame(CapturedFrame *frame, size_t count, size_t blockSize,
                       const uint8_t *weights, size_t weightsSize,
                       size_t payloadSize) {
  *frame = (CapturedFrame){0};
  frame->headerSize = PLANAR_ZIGZAG_TRANSPOSE_PIVCO_HEADER_FIXED +
                      PLANAR_ZIGZAG_TRANSPOSE_PIVCO_HEADER_LANE;
  geozl_st_le64(frame->header, count);
  frame->header[8] = 1;
  geozl_st_le32(frame->header + 9, (uint32_t)count);
  geozl_st_le32(frame->header + 13, 1);
  geozl_st_le32(frame->header + 17, (uint32_t)blockSize);
  frame->header[PLANAR_ZIGZAG_TRANSPOSE_PIVCO_HEADER_FIXED] =
      PLANAR_ZIGZAG_TRANSPOSE_PIVCO_LANE_PIVCO;
  geozl_st_le16(frame->header + PLANAR_ZIGZAG_TRANSPOSE_PIVCO_HEADER_FIXED + 1,
                (uint16_t)weightsSize);
  geozl_st_le32(frame->header + PLANAR_ZIGZAG_TRANSPOSE_PIVCO_HEADER_FIXED + 3,
                (uint32_t)payloadSize);
  memcpy(frame->weights, weights, weightsSize);
  frame->weightsSize = weightsSize;
  frame->bitsSize = payloadSize;
  frame->kind = PIVCO_CAPTURE_KIND_PLANAR_ZIGZAG;
  frame->seen = 1;
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

static int inside(const Bump *bump, const void *memory) {
  return (const uint8_t *)memory >= bump->base &&
         (const uint8_t *)memory < bump->base + bump->used;
}

static int same_plan(const PivcoGpuPlan *a, const PivcoGpuPlan *b) {
  return a->symbolCount == b->symbolCount && a->treeCount == b->treeCount &&
         a->nodeCount == b->nodeCount && a->bitmapCount == b->bitmapCount &&
         a->pivcoBlockCount == b->pivcoBlockCount &&
         a->laneBlockCount == b->laneBlockCount &&
         a->joinCount == b->joinCount && a->laneBytes == b->laneBytes &&
         a->pivcoLaneCount == b->pivcoLaneCount &&
         a->pforLaneCount == b->pforLaneCount &&
         memcmp(a->symbols, b->symbols, a->symbolCount) == 0 &&
         memcmp(a->trees, b->trees, a->treeCount * sizeof(*a->trees)) == 0 &&
         memcmp(a->nodes, b->nodes, a->nodeCount * sizeof(*a->nodes)) == 0 &&
         memcmp(a->pivcoBlocks, b->pivcoBlocks,
                a->pivcoBlockCount * sizeof(*a->pivcoBlocks)) == 0 &&
         memcmp(a->laneBlocks, b->laneBlocks,
                a->laneBlockCount * sizeof(*a->laneBlocks)) == 0 &&
         memcmp(a->joins, b->joins, a->joinCount * sizeof(*a->joins)) == 0;
}

int main(void) {
  enum { count = 4099, blockSize = 1031 };
  static const uint8_t weights[] = {2, 1, 1};
  uint8_t *input = malloc(count);
  if (input == NULL)
    return fail("allocation failed");
  for (size_t i = 0; i < count; ++i)
    input[i] = (uint8_t)(i % 3);

  uint8_t *payload = NULL;
  uint8_t *scratch = NULL;
  size_t payloadSize = 0;
  if (make_lane(&payload, &payloadSize, &scratch, input, count, blockSize,
                weights, sizeof(weights))) {
    free(input);
    free(payload);
    free(scratch);
    return fail("could not encode the test lane");
  }

  CapturedFrame frame;
  make_frame(&frame, count, blockSize, weights, sizeof(weights), payloadSize);
  const GeozlGpuFrameShape shape = {0, count, count, 1};
  PivcoGpuPlan plan;
  if (pivco_gpu_plan_build(&plan, &frame, &shape, 1, payload, payloadSize, 1)) {
    free(scratch);
    free(payload);
    free(input);
    return fail("a valid 1031-byte block was rejected");
  }
  const size_t expectedBlocks = (count + blockSize - 1) / blockSize;
  if (plan.pivcoBlockCount != expectedBlocks ||
      plan.pivcoBlocks[0].length != blockSize ||
      plan.pivcoBlocks[expectedBlocks - 1].length != count % blockSize) {
    pivco_gpu_plan_free(&plan);
    free(scratch);
    free(payload);
    free(input);
    return fail("the block plan has the wrong lengths");
  }
  pivco_gpu_plan_free(&plan);

  CapturedFrame tiny;
  make_frame(&tiny, count, 16, weights, sizeof(weights), payloadSize);
  const int tinyPlanned = pivco_gpu_plan_build(&plan, &tiny, &shape, 1, payload,
                                               payloadSize, 1) == 0;
  pivco_gpu_plan_free(&plan);
  if (tinyPlanned) {
    free(scratch);
    free(payload);
    free(input);
    return fail("16-symbol blocks were accepted");
  }

  enum { frameCount = 4 };
  CapturedFrame frames[frameCount];
  GeozlGpuFrameShape shapes[frameCount];
  for (size_t i = 0; i < frameCount; ++i) {
    frames[i] = frame;
    shapes[i] = (GeozlGpuFrameShape){i * count, count, (uint32_t)count, 1};
  }
  PivcoGpuPlan sequential = {0}, parallel = {0};
  if (pivco_gpu_plan_build(&sequential, frames, shapes, frameCount, payload,
                           payloadSize, 1) ||
      pivco_gpu_plan_build_parallel(&parallel, frames, shapes, frameCount,
                                    payload, payloadSize, 1, frameCount,
                                    NULL)) {
    pivco_gpu_plan_free(&sequential);
    pivco_gpu_plan_free(&parallel);
    free(scratch);
    free(payload);
    free(input);
    return fail("parallel planning rejected valid frames");
  }
  if (!same_plan(&sequential, &parallel)) {
    pivco_gpu_plan_free(&sequential);
    pivco_gpu_plan_free(&parallel);
    free(scratch);
    free(payload);
    free(input);
    return fail("parallel and sequential plans differ");
  }
  pivco_gpu_plan_free(&parallel);

  // The merged arrays land in the caller's memory, which free leaves alone.
  enum { arenaBytes = 1 << 20 };
  Bump bump = {malloc(arenaBytes), 0, arenaBytes};
  const GeozlGpuHostAlloc alloc = {bump_alloc, &bump};
  const int staged =
      bump.base != NULL &&
      pivco_gpu_plan_build_parallel(&parallel, frames, shapes, frameCount,
                                    payload, payloadSize, 1, 2, &alloc) == 0 &&
      parallel.borrowed && same_plan(&sequential, &parallel) &&
      inside(&bump, parallel.symbols) && inside(&bump, parallel.trees) &&
      inside(&bump, parallel.nodes) && inside(&bump, parallel.pivcoBlocks) &&
      inside(&bump, parallel.laneBlocks) && inside(&bump, parallel.joins);
  pivco_gpu_plan_free(&parallel);
  bump.used = 0;
  bump.capacity = 64;
  const int refused =
      bump.base != NULL &&
      pivco_gpu_plan_build_parallel(&parallel, frames, shapes, frameCount,
                                    payload, payloadSize, 1, 2, &alloc) != 0 &&
      parallel.symbols == NULL && parallel.error != NULL;
  pivco_gpu_plan_free(&parallel);
  free(bump.base);
  pivco_gpu_plan_free(&sequential);
  if (!staged || !refused) {
    free(scratch);
    free(payload);
    free(input);
    return fail(!staged ? "a plan was not built in the caller's memory"
                        : "a failed allocation was not reported");
  }

  payload[0] ^= 1;
  if (pivco_gpu_plan_build(&plan, &frame, &shape, 1, payload, payloadSize, 1) ==
      0) {
    pivco_gpu_plan_free(&plan);
    free(scratch);
    free(payload);
    free(input);
    return fail("a bitmap/count mismatch was accepted");
  }

  free(scratch);
  free(payload);
  free(input);
  puts("test_pivco_plan: ok");
  return 0;
}
