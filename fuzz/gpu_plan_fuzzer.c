// Cross-check PFOR and PivCo planners against CPU decoding and buffer bounds.

#include "pfor_plan.h"
#include "pivco_plan.h"

#include "common/endian.h"
#include "openzl/codecs/pivco_huffman/decode_pivco_kernel.h"
#include "openzl/codecs/pivco_huffman/encode_pivco_kernel.h"
#include "pfor/decode_pfor_kernel.h"
#include "pfor/encode_pfor_kernel.h"
#include "pfor/pfor_check.h"
#include "pivco_lanes/pivco_lanes.h"
#include "planar_zigzag_pfor/planar_zigzag_pfor_layout.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MAX_FRAMES 4
#define MAX_COUNT 8192
#define MAX_EDITS 16
#define MAX_ALLOCS 16

typedef struct {
  const uint8_t *data;
  size_t size;
  size_t at;
} Input;

typedef struct {
  uint8_t *data;
  size_t size;
} Payload;

typedef struct {
  void *blocks[MAX_ALLOCS];
  size_t count;
  size_t failAt;
  int failed;
} Tracker;

static uint64_t aligned[MAX_COUNT];
static uint8_t lane[MAX_COUNT];

static unsigned next_u8(Input *in) {
  return in->at < in->size ? in->data[in->at++] : 0;
}

static unsigned next_u16(Input *in) {
  const unsigned low = next_u8(in);
  return low | next_u8(in) << 8;
}

// Frame contents repeat the input, and do not consume it.
static uint8_t data_at(const Input *in, size_t seed, size_t i) {
  return in->data[(seed + i) % in->size];
}

static void *tracker_alloc(void *context, size_t bytes) {
  Tracker *tracker = context;
  if (tracker->count == MAX_ALLOCS ||
      (tracker->failAt != 0 && tracker->count + 1 == tracker->failAt)) {
    tracker->failed = 1;
    return NULL;
  }
  void *memory = malloc(bytes);
  if (memory != NULL)
    tracker->blocks[tracker->count++] = memory;
  return memory;
}

static void tracker_free(Tracker *tracker) {
  for (size_t i = 0; i < tracker->count; ++i)
    free(tracker->blocks[i]);
  tracker->count = 0;
}

// Grows without slack, so ASan sees a read past the end of the stream.
static uint8_t *payload_grow(Payload *payload, size_t extra) {
  uint8_t *grown = realloc(payload->data, payload->size + extra);
  if (grown == NULL)
    abort();
  payload->data = grown;
  payload->size += extra;
  return grown + payload->size - extra;
}

// Supply streams the encoder cannot produce, including invalid bit widths.
static size_t raw_stream(Input *in, Payload *payload) {
  size_t bytes = next_u16(in) % 4096;
  if (bytes > in->size - in->at)
    bytes = in->size - in->at;
  memcpy(payload_grow(payload, bytes + 1), in->data + in->at, bytes);
  payload->size -= 1;
  in->at += bytes;
  return bytes;
}

static void check(int condition) {
  if (!condition)
    abort();
}

static size_t count_of(Input *in, int wide) {
  return wide ? next_u16(in) % MAX_COUNT + 1 : next_u8(in) % 512 + 1;
}

static void check_pfor_plan(const PforGpuPlan *plan, const PforGpuFrame *frames,
                            size_t frameCount, const Payload *payload,
                            size_t elementBytes) {
  size_t task = 0;
  for (size_t i = 0; i < frameCount; ++i) {
    size_t left = frames[i].shape.nbElts;
    size_t at = frames[i].payloadOffset;
    size_t output = frames[i].shape.outOff;
    while (left != 0) {
      check(task < plan->realBlockCount);
      const GpuPforBlock *block = &plan->blocks[task++];
      const size_t count = left < GEOZL_PFOR_BLOCK ? left : GEOZL_PFOR_BLOCK;
      check(block->payloadOffset == at && block->outputBase == output &&
            block->count == count && block->pad == 0);
      PforGpuBlockInfo info;
      check(pfor_gpu_block_info(payload->data + at, payload->size - at,
                                elementBytes, &info) == 0);
      at += info.bytes;
      output += count;
      left -= count;
    }
    check(at == frames[i].payloadOffset + frames[i].payloadSize);
  }
  check(task == plan->realBlockCount);
  check(plan->blockCount == (task + 3) / 4 * 4);
}

static void check_pfor_same(const PforGpuPlan *a, const PforGpuPlan *b) {
  check(a->blockCount == b->blockCount &&
        a->realBlockCount == b->realBlockCount &&
        memcmp(a->blocks, b->blocks, a->blockCount * sizeof(*a->blocks)) == 0);
}

static void fuzz_pfor(Input *in, unsigned flags, size_t elementBytes) {
  const size_t frameCount = ((flags >> 3) & 3u) + 1;
  const size_t threads = (flags >> 5) % 3 + 1;
  PforGpuFrame frames[MAX_FRAMES];
  Payload payload = {0};
  int pristine = 1;
  size_t output = 0;

  for (size_t i = 0; i < frameCount; ++i) {
    const unsigned mode = next_u8(in);
    const size_t count = count_of(in, mode & 1u);
    const size_t seed = next_u8(in);
    const unsigned bits = next_u8(in) % (unsigned)(elementBytes * 8 + 1);
    const uint64_t mask = (1ull << bits) - 1;
    for (size_t j = 0; j < count; ++j) {
      const uint64_t value = (uint64_t)(data_at(in, seed, j * 2) |
                                        data_at(in, seed, j * 2 + 1) << 8) &
                             mask;
      if (elementBytes == 1)
        ((uint8_t *)aligned)[j] = (uint8_t)value;
      else
        ((uint16_t *)aligned)[j] = (uint16_t)value;
    }
    const size_t base = payload.size;
    size_t written = 0;
    if ((mode & 4u) != 0) {
      pristine = 0;
      written = raw_stream(in, &payload);
    } else {
      const size_t bound = pfor_bound(count, elementBytes);
      check(bound != 0);
      uint8_t *packed = payload_grow(&payload, bound);
      check(pfor_encode(packed, bound, &written, aligned, count,
                        elementBytes) == 0);
      payload.size = base + written;
    }

    frames[i] = (PforGpuFrame){
        .shape = {output, count, count, 1},
        .payloadOffset = base,
        .payloadSize = written,
        .transform = PFOR_GPU_ID,
    };
    output += count;
  }
  if (payload.size == 0)
    payload_grow(&payload, 1);
  uint8_t *exact = malloc(payload.size);
  check(exact != NULL);
  memcpy(exact, payload.data, payload.size);
  free(payload.data);
  payload.data = exact;

  const unsigned edits = next_u8(in) % (MAX_EDITS + 1);
  for (unsigned e = 0; e < edits; ++e) {
    const unsigned area = next_u8(in);
    const unsigned position = next_u16(in);
    const unsigned change = next_u8(in);
    if (change == 0)
      continue;
    pristine = 0;
    if ((area & 3u) == 3u) {
      PforGpuFrame *frame = &frames[position % frameCount];
      const size_t step = change >> 1;
      if ((change & 1u) == 0) {
        if (frame->payloadSize > step)
          frame->payloadSize -= step;
      } else if ((area & 4u) != 0) {
        if (frame->shape.nbElts + step <= MAX_COUNT)
          frame->shape.nbElts += step;
      } else if (frame->shape.nbElts > step) {
        frame->shape.nbElts -= step;
      }
    } else {
      payload.data[position % payload.size] ^= (uint8_t)change;
    }
  }

  size_t blockCounts[MAX_FRAMES];
  size_t totalBlocks = 0;
  for (size_t i = 0; i < frameCount; ++i) {
    blockCounts[i] =
        (frames[i].shape.nbElts + GEOZL_PFOR_BLOCK - 1) / GEOZL_PFOR_BLOCK;
    totalBlocks += blockCounts[i];
  }
  GpuPforBlock *blocks = calloc(totalBlocks + 4, sizeof(*blocks));
  check(blocks != NULL);
  int framesFailed = 0;
  size_t task = 0;
  size_t exceptions = 0;
  for (size_t i = 0; i < frameCount; ++i) {
    PforGpuFrameTally tally;
    if (pfor_gpu_plan_frame(blocks + task, &frames[i], payload.data,
                            elementBytes, &tally) != 0) {
      check(tally.error != NULL);
      framesFailed = 1;
    } else {
      exceptions += tally.exceptionCount;
    }
    task += blockCounts[i];
  }

  PforGpuPlan whole;
  const int wholeFailed = pfor_gpu_plan_build(
      &whole, frames, frameCount, payload.data, payload.size, elementBytes);
  check(wholeFailed == framesFailed);
  check(wholeFailed ? whole.error != NULL : whole.exceptionCount == exceptions);
  check(!pristine || !wholeFailed);

  Tracker tracker = {.failAt = next_u8(in) % 3};
  const GeozlGpuHostAlloc alloc = {tracker_alloc, &tracker};
  PforGpuPlan parallel;
  const int parallelFailed = pfor_gpu_plan_build_parallel(
      &parallel, frames, frameCount, payload.data, payload.size, elementBytes,
      threads, (flags & 0x80u) ? &alloc : NULL);
  check(parallelFailed == wholeFailed || tracker.failed);

  if (!wholeFailed) {
    check_pfor_plan(&whole, frames, frameCount, &payload, elementBytes);
    PforGpuPlan wrapped;
    pfor_gpu_plan_wrap(&wrapped, blocks, totalBlocks);
    check_pfor_same(&wrapped, &whole);
    pfor_gpu_plan_free(&wrapped);
    if (!parallelFailed)
      check_pfor_same(&parallel, &whole);

    for (size_t i = 0; i < frameCount; ++i) {
      const size_t bytes = frames[i].shape.nbElts * elementBytes;
      uint8_t *decoded = malloc(bytes);
      check(decoded != NULL);
      check(pfor_decode(decoded, frames[i].shape.nbElts, elementBytes,
                        payload.data + frames[i].payloadOffset,
                        frames[i].payloadSize) == 0);
      free(decoded);
    }
  }
  pfor_gpu_plan_free(&parallel);
  tracker_free(&tracker);
  pfor_gpu_plan_free(&whole);
  free(blocks);
  free(payload.data);
}

// Independent band decomposition for the oracle.
static unsigned ref_band(size_t rowsLeft) {
  unsigned h = 16;
  while (h > rowsLeft)
    h >>= 1;
  return h;
}

// Compare the band planner with an independent geometry walk.
static void fuzz_bands(Input *in, unsigned flags) {
  const size_t frameCount = ((flags >> 3) & 3u) + 1;
  PforGpuFrame frames[MAX_FRAMES];
  size_t ids[MAX_FRAMES];
  size_t output = next_u16(in);
  int fits = 1;
  for (size_t i = 0; i < frameCount; ++i) {
    const unsigned mode = next_u8(in);
    size_t width = next_u16(in) % 5000 + 1;
    size_t rows = next_u8(in) % 48 + 1;
    const uint32_t planes = next_u8(in) % 3 + 1;
    if ((mode & 8u) != 0) {
      width = 1;
      rows = next_u16(in) % 4096 + 1;
    }
    size_t payloadOffset = next_u16(in);
    const size_t headers = 3 * geozl_pzp_plane_blocks(width, rows) * planes;
    size_t payloadSize = (mode & 128u) ? headers - 1 : headers + next_u16(in);
    if ((mode & 16u) != 0)
      payloadSize = UINT32_MAX - (next_u16(in) << 8);
    if ((mode & 32u) != 0)
      payloadOffset = UINT32_MAX - payloadSize + 1 - next_u8(in) % 2;
    if ((mode & 64u) != 0)
      output = UINT32_MAX - (size_t)next_u16(in) * 16;
    frames[i] = (PforGpuFrame){
        .shape = {output, width * rows * planes, width, planes},
        .payloadOffset = payloadOffset,
        .payloadSize = payloadSize,
        .transform = (mode & 1u) ? PFOR_GPU_ID : PFOR_GPU_PLANAR_ZIGZAG,
    };
    if ((mode & 2u) != 0)
      frames[i].shape.nbElts -= next_u8(in) % 3;
    if ((mode & 4u) != 0)
      frames[i].shape.width = width + next_u8(in) % 3;
    ids[i] = next_u16(in);
    if (frames[i].transform == PFOR_GPU_PLANAR_ZIGZAG)
      fits &= pfor_gpu_band_frame_fits(&frames[i]);
    output += frames[i].shape.nbElts;
  }

  Tracker tracker = {.failAt = next_u8(in) % 3};
  const GeozlGpuHostAlloc alloc = {tracker_alloc, &tracker};
  const size_t *frameIds = (flags & 0x20u) ? ids : NULL;
  PforGpuBandPlan plan;
  const int failed = pfor_gpu_band_plan_build(
      &plan, frames, frameCount, frameIds, (flags & 0x80u) ? &alloc : NULL);
  if (failed) {
    check(plan.error != NULL && plan.frames == NULL && plan.strips == NULL &&
          plan.chunks == NULL);
    check(!fits || tracker.failed);
    tracker_free(&tracker);
    return;
  }
  check(fits);

  size_t f = 0, s = 0, block = 0;
  for (size_t i = 0; i < frameCount; ++i) {
    const PforGpuFrame *frame = &frames[i];
    if (frame->transform != PFOR_GPU_PLANAR_ZIGZAG)
      continue;
    const uint32_t id = (uint32_t)(frameIds == NULL ? i : frameIds[i]);
    const size_t width = frame->shape.width;
    const size_t planeElts = frame->shape.nbElts / frame->shape.planes;
    const size_t rows = planeElts / width;
    const size_t first = block;
    for (size_t p = 0; p < frame->shape.planes; ++p) {
      for (size_t y0 = 0; y0 < rows;) {
        const unsigned h = ref_band(rows - y0);
        for (size_t x0 = 0; x0 < width; x0 += 1024) {
          const size_t columns = width - x0 < 1024 ? width - x0 : 1024;
          check(s < plan.stripCount);
          const GpuPforStrip *strip = &plan.strips[s++];
          check(strip->firstBlock == block && strip->height == h &&
                strip->blockCount == (columns * h + 255) / 256 &&
                strip->outputBase ==
                    frame->shape.outOff + p * planeElts + y0 * width + x0 &&
                strip->width == width && strip->columns == columns &&
                strip->frame == id && strip->pad == 0);
          block += strip->blockCount;
        }
        y0 += h;
      }
    }
    check(f < plan.frameCount);
    const GpuPforBandFrame *planned = &plan.frames[f++];
    check(planned->payloadOffset == frame->payloadOffset &&
          planned->payloadSize == frame->payloadSize &&
          planned->firstBlock == first &&
          planned->blockCount == block - first && planned->frame == id &&
          block - first ==
              geozl_pzp_plane_blocks(width, rows) * frame->shape.planes);
  }
  check(plan.blockCount == block);
  // Chunks must cover each frame exactly once.
  size_t chunk = 0;
  for (size_t i = 0; i < f; ++i) {
    const size_t blocks = plan.frames[i].blockCount;
    const size_t size = blocks > 256 * 1024 ? (blocks + 1023) / 1024 : 256;
    const size_t firstChunk = chunk;
    for (size_t first = 0; first < blocks; first += size) {
      check(chunk < plan.chunkCount);
      const GpuPforBlockChunk *c = &plan.chunks[chunk++];
      check(c->frame == i && c->first == first && c->firstChunk == firstChunk &&
            c->count == (blocks - first < size ? blocks - first : size));
    }
  }
  check(plan.chunkCount == (chunk + 3) / 4 * 4);
  for (; chunk < plan.chunkCount; ++chunk)
    check(plan.chunks[chunk].count == 0);
  check(plan.frameCount == (f + 3) / 4 * 4 &&
        plan.stripCount == (s + 3) / 4 * 4);
  for (; f < plan.frameCount; ++f)
    check(plan.frames[f].blockCount == 0);
  for (; s < plan.stripCount; ++s)
    check(plan.strips[s].blockCount == 0);
  pfor_gpu_band_plan_free(&plan);
  tracker_free(&tracker);
}

// Split leaves to build a complete prefix code with input-selected depths.
static size_t build_weights(Input *in, uint8_t symbols[], uint8_t weights[],
                            size_t *weightsSize, int *tableLog) {
  uint8_t lengths[ZL_PIVCO_MAX_SYMBOLS] = {0};
  size_t leaves = 1;
  const unsigned splits = next_u8(in);
  for (unsigned s = 0; s < splits && leaves < ZL_PIVCO_MAX_SYMBOLS; ++s) {
    const size_t leaf = next_u8(in) % leaves;
    if (lengths[leaf] >= ZL_PIVCO_MAX_TABLE_LOG)
      continue;
    ++lengths[leaf];
    lengths[leaves++] = lengths[leaf];
  }
  unsigned maxLength = 0;
  for (size_t i = 0; i < leaves; ++i) {
    if (lengths[i] > maxLength)
      maxLength = lengths[i];
  }
  const size_t stride = 1 + next_u8(in) % 3;
  *weightsSize = (leaves - 1) * stride + 1;
  if (*weightsSize > ZL_PIVCO_MAX_SYMBOLS)
    *weightsSize = leaves;
  const size_t step = *weightsSize == leaves ? 1 : stride;
  memset(weights, 0, *weightsSize);
  for (size_t i = 0; i < leaves; ++i) {
    symbols[i] = (uint8_t)(i * step);
    weights[i * step] = (uint8_t)(maxLength + 1 - lengths[i]);
  }
  *tableLog = ZL_PivCoHuffman_computeTableLog(weights, *weightsSize);
  check(*tableLog == (int)maxLength);
  return leaves;
}

static void make_pivco_frame(Input *in, Payload *payload, CapturedFrame *frame,
                             GeozlGpuFrameShape *shape, size_t elementBytes,
                             size_t output, int *pristine) {
  static const size_t kBlocks[4] = {0, 1024, 4096, 32768};
  const unsigned mode = next_u8(in);
  const size_t count = count_of(in, mode & 1u);
  const int planar = (mode & 2u) != 0;
  const size_t picked = kBlocks[(mode >> 2) & 3u];
  const size_t blockSize = picked != 0 ? picked : count;
  const size_t fixedHeader = planar ? PLANAR_ZIGZAG_LANES_HEADER_FIXED
                                    : LANES_HEADER_FIXED;

  *frame = (CapturedFrame){0};
  frame->kind =
      planar ? PIVCO_CAPTURE_KIND_PLANAR_ZIGZAG : PIVCO_CAPTURE_KIND_ID;
  frame->headerSize = fixedHeader + elementBytes * GEOZL_PIVCO_LANE_HEADER_SIZE;
  frame->bitsOffset = payload->size;
  geozl_st_le64(frame->header, count);
  frame->header[8] = (uint8_t)elementBytes;
  if (planar) {
    geozl_st_le32(frame->header + 9, (uint32_t)count);
    geozl_st_le32(frame->header + 13, 1);
    geozl_st_le32(frame->header + 17, (uint32_t)blockSize);
  } else {
    geozl_st_le32(frame->header + 9, (uint32_t)blockSize);
  }

  for (size_t laneIndex = 0; laneIndex < elementBytes; ++laneIndex) {
    uint8_t symbols[ZL_PIVCO_MAX_SYMBOLS];
    uint8_t weights[ZL_PIVCO_MAX_SYMBOLS];
    size_t weightsSize = 0;
    int tableLog = 0;
    const size_t leaves =
        build_weights(in, symbols, weights, &weightsSize, &tableLog);
    const size_t seed = next_u8(in);
    for (size_t j = 0; j < count; ++j)
      lane[j] = symbols[data_at(in, seed, j) % leaves];

    uint8_t *laneHeader = frame->header + fixedHeader +
                          laneIndex * GEOZL_PIVCO_LANE_HEADER_SIZE;
    const size_t base = payload->size;
    size_t used;
    const unsigned laneMode = next_u8(in);
    if ((laneMode & 3u) == 0) {
      if ((laneMode & 4u) != 0) {
        *pristine = 0;
        used = raw_stream(in, payload);
      } else {
        const size_t bound = pfor_bound(count, 1);
        check(bound != 0);
        uint8_t *bits = payload_grow(payload, bound);
        check(pfor_encode(bits, bound, &used, lane, count, 1) == 0);
      }
      laneHeader[0] = GEOZL_PIVCO_LANE_PFOR;
      weightsSize = 0;
    } else {
      const size_t bound = ZL_PivCoHuffmanEncode_bound(count, blockSize);
      const size_t scratchSize =
          ZL_PivCoHuffmanEncode_scratchElements(count, blockSize);
      uint8_t *scratch = malloc(scratchSize);
      check(scratch != NULL);
      uint8_t *bits = payload_grow(payload, bound);
      used = ZL_PivCoHuffman_encode(bits, bound, scratch, scratchSize, weights,
                                    weightsSize, tableLog, lane, count,
                                    blockSize, NULL);
      free(scratch);
      check(used != SIZE_MAX);
      laneHeader[0] = GEOZL_PIVCO_LANE_PIVCO;
      memcpy(frame->weights + frame->weightsSize, weights, weightsSize);
    }
    payload->size = base + used;
    geozl_st_le16(laneHeader + 1, (uint16_t)weightsSize);
    geozl_st_le32(laneHeader + 3, (uint32_t)used);
    frame->weightsSize += weightsSize;
    frame->bitsSize += used;
  }
  *shape = (GeozlGpuFrameShape){output, count, count, 1};
}

static void check_pivco_plan(const PivcoGpuPlan *plan, const uint8_t *payload,
                             size_t payloadSize, size_t elementBytes) {
  for (size_t b = 0; b < plan->pivcoBlockCount; ++b) {
    const GpuPivcoBlock *block = &plan->pivcoBlocks[b];
    check(block->nodeCount != 0 &&
          (size_t)block->nodeBase + block->nodeCount <= plan->nodeCount &&
          (size_t)block->treeBase + block->nodeCount <= plan->treeCount &&
          (size_t)block->symbolBase + ZL_PIVCO_MAX_SYMBOLS <=
              plan->symbolCount &&
          (size_t)block->outBase + block->length <= plan->laneBytes);
    check(plan->nodes[block->nodeBase].count == block->length &&
          plan->nodes[block->nodeBase].output == block->outBase);
    for (uint32_t i = 0; i < block->nodeCount; ++i) {
      const GpuTreeNode *tree = &plan->trees[block->treeBase + i];
      const GpuNodeBits *node = &plan->nodes[block->nodeBase + i];
      check((size_t)node->output + node->count <=
            (size_t)block->outBase + block->length);
      if ((tree->leaf & (1u << 16)) != 0) {
        const size_t depth = (tree->leaf >> 8) & 0xFFu;
        if (depth != 0)
          check((size_t)node->offset + (node->count * depth + 7) / 8 <=
                payloadSize);
      } else {
        const size_t left = tree->links & 0xFFFFu;
        const size_t right = tree->links >> 16;
        check(left < block->nodeCount && right < block->nodeCount);
        check((size_t)node->offset + (node->count + 7) / 8 <= payloadSize);
        check((size_t)plan->nodes[block->nodeBase + left].count +
                  plan->nodes[block->nodeBase + right].count ==
              node->count);
      }
    }
  }
  check(plan->laneBlockCount % 4 == 0);
  for (size_t i = 0; i < plan->laneBlockCount; ++i) {
    PforGpuBlockInfo info;
    const size_t at = plan->laneBlocks[i].payloadOffset;
    check(at < payloadSize && pfor_gpu_block_info(payload + at, payloadSize - at,
                                                  1, &info) == 0);
    check((size_t)plan->laneBlocks[i].outputBase + GEOZL_PFOR_BLOCK <=
          plan->laneBytes + 256);
  }
  for (size_t i = 0; i < plan->joinCount; ++i) {
    const GpuJoinTask *join = &plan->joins[i];
    check(join->start < join->count);
    check((size_t)join->laneBase + join->laneStride * elementBytes <=
          plan->laneBytes);
    check(join->laneStride >= join->count);
  }
}

static void check_pivco_same(const PivcoGpuPlan *a, const PivcoGpuPlan *b) {
  check(a->symbolCount == b->symbolCount && a->treeCount == b->treeCount &&
        a->nodeCount == b->nodeCount && a->bitmapCount == b->bitmapCount &&
        a->pivcoBlockCount == b->pivcoBlockCount &&
        a->laneBlockCount == b->laneBlockCount &&
        a->joinCount == b->joinCount && a->laneBytes == b->laneBytes &&
        a->pivcoLaneCount == b->pivcoLaneCount &&
        a->pforLaneCount == b->pforLaneCount);
  check(a->symbolCount == 0 ||
        memcmp(a->symbols, b->symbols, a->symbolCount) == 0);
  check(a->treeCount == 0 ||
        memcmp(a->trees, b->trees, a->treeCount * sizeof(*a->trees)) == 0);
  check(a->nodeCount == 0 ||
        memcmp(a->nodes, b->nodes, a->nodeCount * sizeof(*a->nodes)) == 0);
  check(a->pivcoBlockCount == 0 ||
        memcmp(a->pivcoBlocks, b->pivcoBlocks,
               a->pivcoBlockCount * sizeof(*a->pivcoBlocks)) == 0);
  check(a->laneBlockCount == 0 ||
        memcmp(a->laneBlocks, b->laneBlocks,
               a->laneBlockCount * sizeof(*a->laneBlocks)) == 0);
  check(a->joinCount == 0 ||
        memcmp(a->joins, b->joins, a->joinCount * sizeof(*a->joins)) == 0);
}

// The CPU decoder must also accept every frame the GPU planner accepts.
static void check_pivco_cpu(const CapturedFrame *frame,
                            const GeozlGpuFrameShape *shape,
                            const uint8_t *payload, size_t elementBytes) {
  const int planar = frame->kind == PIVCO_CAPTURE_KIND_PLANAR_ZIGZAG;
  const size_t fixedHeader = planar ? PLANAR_ZIGZAG_LANES_HEADER_FIXED
                                    : LANES_HEADER_FIXED;
  const size_t blockSize = geozl_ld_le32(frame->header + (planar ? 17 : 9));
  const size_t count = shape->nbElts;
  const size_t scratchSize =
      ZL_PivCoHuffmanDecode_scratchBytes(count, blockSize);
  uint8_t *scratch = malloc(scratchSize);
  uint8_t *decoded = malloc(count);
  check(scratch != NULL && decoded != NULL);
  size_t weightsOffset = 0;
  size_t bitsOffset = frame->bitsOffset;
  for (size_t laneIndex = 0; laneIndex < elementBytes; ++laneIndex) {
    const uint8_t *laneHeader = frame->header + fixedHeader +
                                laneIndex * GEOZL_PIVCO_LANE_HEADER_SIZE;
    const size_t weightsSize = geozl_ld_le16(laneHeader + 1);
    const size_t bitsSize = geozl_ld_le32(laneHeader + 3);
    if (laneHeader[0] == GEOZL_PIVCO_LANE_PFOR)
      check(pfor_decode(decoded, count, 1, payload + bitsOffset, bitsSize) ==
            0);
    else
      check(ZL_PivCoHuffman_decode(decoded, count, scratch, scratchSize,
                                   frame->weights + weightsOffset, weightsSize,
                                   payload + bitsOffset, bitsSize, blockSize,
                                   NULL));
    weightsOffset += weightsSize;
    bitsOffset += bitsSize;
  }
  free(scratch);
  free(decoded);
}

static void fuzz_pivco(Input *in, unsigned flags, size_t elementBytes) {
  static CapturedFrame frames[MAX_FRAMES];
  const size_t frameCount = ((flags >> 3) & 3u) + 1;
  const size_t threads = (flags >> 5) % 3 + 1;
  GeozlGpuFrameShape shapes[MAX_FRAMES];
  Payload payload = {0};
  int pristine = 1;
  size_t output = 0;

  for (size_t i = 0; i < frameCount; ++i) {
    make_pivco_frame(in, &payload, &frames[i], &shapes[i], elementBytes, output,
                     &pristine);
    output += shapes[i].nbElts;
  }
  if (payload.size == 0)
    payload_grow(&payload, 1);
  uint8_t *exact = malloc(payload.size);
  check(exact != NULL);
  memcpy(exact, payload.data, payload.size);
  free(payload.data);
  payload.data = exact;

  const unsigned edits = next_u8(in) % (MAX_EDITS + 1);
  for (unsigned e = 0; e < edits; ++e) {
    const unsigned area = next_u8(in);
    const unsigned position = next_u16(in);
    const unsigned change = next_u8(in);
    if (change == 0)
      continue;
    pristine = 0;
    CapturedFrame *frame = &frames[(area >> 2) % frameCount];
    switch (area & 3u) {
    case 0:
      frame->header[position % frame->headerSize] ^= (uint8_t)change;
      break;
    case 1:
      if (frame->weightsSize != 0)
        frame->weights[position % frame->weightsSize] ^= (uint8_t)change;
      break;
    case 2:
      payload.data[position % payload.size] ^= (uint8_t)change;
      break;
    default:
      if (position & 1u) {
        if (frame->weightsSize + change <= sizeof(frame->weights))
          frame->weightsSize += change;
      } else if (frame->bitsSize >= change) {
        frame->bitsSize -= change;
      }
      break;
    }
  }

  // Use the possibly mutated header shape in half the cases.
  if ((next_u8(in) & 1u) != 0) {
    for (size_t i = 0; i < frameCount; ++i) {
      if (frames[i].kind == PIVCO_CAPTURE_KIND_PLANAR_ZIGZAG) {
        shapes[i].width = geozl_ld_le32(frames[i].header + 9);
        shapes[i].planes = geozl_ld_le32(frames[i].header + 13);
      }
    }
  }

  PivcoGpuPlan parts[MAX_FRAMES];
  int partsFailed = 0;
  for (size_t i = 0; i < frameCount; ++i) {
    if (pivco_gpu_plan_build(&parts[i], &frames[i], &shapes[i], 1, payload.data,
                             payload.size, elementBytes) != 0) {
      check(parts[i].error != NULL);
      partsFailed = 1;
    }
  }

  PivcoGpuPlan whole;
  const int wholeFailed =
      pivco_gpu_plan_build(&whole, frames, shapes, frameCount, payload.data,
                           payload.size, elementBytes);
  check(wholeFailed == partsFailed);
  check(!wholeFailed || whole.error != NULL);
  check(!pristine || !wholeFailed);

  Tracker tracker = {.failAt = next_u8(in) % 14};
  const GeozlGpuHostAlloc alloc = {tracker_alloc, &tracker};
  const GeozlGpuHostAlloc *allocator = (flags & 0x80u) ? &alloc : NULL;
  PivcoGpuPlan parallel;
  const int parallelFailed = pivco_gpu_plan_build_parallel(
      &parallel, frames, shapes, frameCount, payload.data, payload.size,
      elementBytes, threads, allocator);
  check(parallelFailed == wholeFailed || tracker.failed);

  if (!wholeFailed) {
    check_pivco_plan(&whole, payload.data, payload.size, elementBytes);
    if (!parallelFailed)
      check_pivco_same(&parallel, &whole);
    PivcoGpuPlan merged;
    if (pivco_gpu_plan_merge(&merged, parts, frameCount, allocator) == 0)
      check_pivco_same(&merged, &whole);
    else
      check(tracker.failed && merged.error != NULL);
    pivco_gpu_plan_free(&merged);
    for (size_t i = 0; i < frameCount; ++i)
      check_pivco_cpu(&frames[i], &shapes[i], payload.data, elementBytes);
  }

  pivco_gpu_plan_free(&parallel);
  tracker_free(&tracker);
  pivco_gpu_plan_free(&whole);
  for (size_t i = 0; i < frameCount; ++i)
    pivco_gpu_plan_free(&parts[i]);
  free(payload.data);
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  if (size < 2)
    return 0;
  Input in = {data + 1, size - 1, 0};
  const size_t elementBytes = (data[0] & 2u) ? 2 : 1;
  if (data[0] & 1u)
    fuzz_pivco(&in, data[0], elementBytes);
  else if (data[0] & 4u)
    fuzz_bands(&in, data[0]);
  else
    fuzz_pfor(&in, data[0], elementBytes);
  return 0;
}
