#include "pivco_plan.h"

#include "common/endian.h"
#include "pfor/pfor_check.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(sizeof(GpuTreeNode) == 8, "GpuTreeNode must match Slang");
_Static_assert(sizeof(GpuNodeBits) == 12, "GpuNodeBits must match Slang");
_Static_assert(sizeof(GpuPivcoBlock) == 24,
               "GpuPivcoBlock must match Slang");
_Static_assert(sizeof(GpuLaneBlock) == 8, "GpuLaneBlock must match Slang");
_Static_assert(sizeof(GpuJoinTask) == 16, "GpuJoinTask must match Slang");

typedef struct {
  GpuTreeNode nodes[ZL_PIVCO_MAX_TREE_NODES];
  uint8_t bothChildrenConstant[ZL_PIVCO_MAX_TREE_NODES];
  size_t count;
} Topology;

typedef struct {
  const uint8_t *data;
  size_t bit;
  size_t sizeBits;
} BitCursor;

static int reserve_array(void **array, size_t *capacity, size_t used,
                         size_t extra, size_t elementBytes) {
  if (extra > SIZE_MAX - used)
    return 1;
  const size_t needed = used + extra;
  if (needed <= *capacity)
    return 0;

  size_t next = *capacity ? *capacity : 1024;
  while (next < needed) {
    if (next > SIZE_MAX / 2) {
      next = needed;
      break;
    }
    next *= 2;
  }
  if (next > SIZE_MAX / elementBytes)
    return 1;
  void *grown = realloc(*array, next * elementBytes);
  if (grown == NULL)
    return 1;
  *array = grown;
  *capacity = next;
  return 0;
}

#define RESERVE(plan, member, capacity, used, extra)                           \
  reserve_array((void **)&(plan)->member, &(capacity), (used), (extra),        \
                sizeof *(plan)->member)

static int as_u32(size_t value, uint32_t *out) {
  if (value > UINT32_MAX)
    return 1;
  *out = (uint32_t)value;
  return 0;
}

// Builds a lane's tree in the same pre-order used by its bitstream.
static int topology_build(Topology *topology, const ZL_PivCoHuffmanTree *tree,
                          size_t level, size_t first, size_t end) {
  if (topology->count >= ZL_PIVCO_MAX_TREE_NODES || first >= end ||
      first > UINT8_MAX || level > UINT8_MAX)
    return -1;
  const size_t current = topology->count++;
  topology->bothChildrenConstant[current] = 0;
  if (ZL_PivCoHuffmanTree_rangeIsLeaf(tree, first, end)) {
    topology->nodes[current] = (GpuTreeNode){
        0, (uint32_t)first | ((uint32_t)tree->rankToFlatDepth[first] << 8) |
               (1u << 16)};
    return (int)current;
  }

  const size_t split = ZL_PivCoHuffmanTree_splitRank(tree, level, first, end);
  if (split <= first || split >= end)
    return -1;
  topology->bothChildrenConstant[current] =
      ZL_PivCoHuffmanTree_rangeIsConstantLeaf(tree, first, split) &&
      ZL_PivCoHuffmanTree_rangeIsConstantLeaf(tree, split, end);
  const int left = topology_build(topology, tree, level + 1, first, split);
  const int right = topology_build(topology, tree, level + 1, split, end);
  if (left < 0 || right < 0)
    return -1;
  topology->nodes[current] =
      (GpuTreeNode){(uint32_t)left | ((uint32_t)right << 16), (uint32_t)first};
  return (int)current;
}

static int cursor_read(BitCursor *cursor, unsigned bits, uint32_t *value) {
  if (bits > 32 || cursor->bit > cursor->sizeBits ||
      bits > cursor->sizeBits - cursor->bit)
    return 1;
  uint32_t result = 0;
  for (unsigned k = 0; k < bits; ++k, ++cursor->bit)
    result |=
        (uint32_t)((cursor->data[cursor->bit >> 3] >> (cursor->bit & 7)) & 1u)
        << k;
  *value = result;
  return 0;
}

static int cursor_align_byte(BitCursor *cursor) {
  if (cursor->bit > SIZE_MAX - 7)
    return 1;
  cursor->bit = (cursor->bit + 7) & ~(size_t)7;
  return cursor->bit > cursor->sizeBits;
}

// Records every node bitmap in one block and reserves its rank-directory
// words. OpenZL has already validated this stream; the bounds below keep this
// planner safe if it is later reused without that prerequisite.
static int walk_block(const Topology *topology, size_t node, size_t count,
                      BitCursor *cursor, size_t payloadOffset,
                      GpuNodeBits *output, size_t *rankWords) {
  if (node >= topology->count || count > UINT32_MAX)
    return 1;
  const GpuTreeNode treeNode = topology->nodes[node];
  output[node] = (GpuNodeBits){0, (uint32_t)count, 0};

  if (treeNode.leaf >> 16) {
    const unsigned depth = (treeNode.leaf >> 8) & 0xFFu;
    if (depth == 0)
      return 0;
    if (cursor_align_byte(cursor) != 0 || count > SIZE_MAX / depth ||
        count * depth > cursor->sizeBits - cursor->bit ||
        payloadOffset > UINT32_MAX - cursor->bit / 8)
      return 1;
    output[node].offset = (uint32_t)(payloadOffset + cursor->bit / 8);
    cursor->bit += count * depth;
    return 0;
  }

  if (cursor_align_byte(cursor) != 0 ||
      count > cursor->sizeBits - cursor->bit ||
      payloadOffset > UINT32_MAX - cursor->bit / 8 || *rankWords > UINT32_MAX)
    return 1;
  output[node].offset = (uint32_t)(payloadOffset + cursor->bit / 8);
  output[node].ranks = (uint32_t)*rankWords;
  const size_t words = (count + 31) / 32;
  if (words > SIZE_MAX - *rankWords)
    return 1;
  *rankWords += words;
  cursor->bit += count;

  size_t ones = 0;
  if (!topology->bothChildrenConstant[node]) {
    unsigned countBits = 0;
    while (((size_t)1 << countBits) < count + 1)
      ++countBits;
    uint32_t encodedOnes = 0;
    if (cursor_read(cursor, countBits, &encodedOnes) != 0 ||
        encodedOnes > count)
      return 1;
    ones = encodedOnes;
  }

  const size_t left = treeNode.links & 0xFFFFu;
  const size_t right = treeNode.links >> 16;
  return walk_block(topology, left, count - ones, cursor, payloadOffset, output,
                    rankWords) ||
         walk_block(topology, right, ones, cursor, payloadOffset, output,
                    rankWords);
}

void pivco_gpu_plan_free(PivcoGpuPlan *plan) {
  if (plan == NULL)
    return;
  free(plan->symbols);
  free(plan->trees);
  free(plan->nodes);
  free(plan->internalNodes);
  free(plan->pivcoBlocks);
  free(plan->laneBlocks);
  free(plan->joins);
  *plan = (PivcoGpuPlan){0};
}

static int fail(PivcoGpuPlan *plan, const char *message, size_t frame,
                size_t lane) {
  if (lane == SIZE_MAX)
    fprintf(stderr, "%s in frame %zu\n", message, frame);
  else
    fprintf(stderr, "%s in lane %zu of frame %zu\n", message, lane, frame);
  pivco_gpu_plan_free(plan);
  return 1;
}

int pivco_gpu_plan_build(PivcoGpuPlan *plan, const CapturedFrame *frames,
                         const Shape *shapes, size_t frameCount,
                         const uint8_t *payload, size_t payloadSize,
                         size_t elementBytes) {
  if (plan == NULL || frames == NULL || shapes == NULL || payload == NULL ||
      frameCount == 0 || (elementBytes != 1 && elementBytes != 2))
    return 1;
  *plan = (PivcoGpuPlan){0};

  size_t symbolCapacity = 0, treeCapacity = 0, nodeCapacity = 0;
  size_t internalCapacity = 0, pivcoCapacity = 0, laneCapacity = 0;
  plan->joins = calloc(frameCount, sizeof(*plan->joins));
  if (plan->joins == NULL)
    return 1;

  Topology topology;
  for (size_t frameIndex = 0; frameIndex < frameCount; ++frameIndex) {
    const CapturedFrame *captured = &frames[frameIndex];
    const Shape *shape = &shapes[frameIndex];
    const uint8_t *header = captured->header;
    const uint64_t encodedCount = geozl_ld_le64(header);
    if (encodedCount > SIZE_MAX)
      return fail(plan, "element count exceeds the host size", frameIndex,
                  SIZE_MAX);
    const size_t count = (size_t)encodedCount;
    const size_t blockSize = geozl_ld_le32(header + 17);
    const size_t expectedHeader =
        PLANAR_ZIGZAG_PIVCO_HEADER_FIXED +
        elementBytes * PLANAR_ZIGZAG_PIVCO_HEADER_LANE;
    if (captured->headerSize != expectedHeader || header[8] != elementBytes ||
        count != shape->nbElts || geozl_ld_le32(header + 9) != shape->width ||
        geozl_ld_le32(header + 13) != shape->planes || blockSize == 0 ||
        blockSize > ZL_PIVCO_MAX_BLOCK_SIZE)
      return fail(plan, "unexpected codec header", frameIndex, SIZE_MAX);
    if (captured->bitsOffset > payloadSize ||
        captured->bitsSize > payloadSize - captured->bitsOffset)
      return fail(plan, "captured payload exceeds its buffer", frameIndex,
                  SIZE_MAX);
    const size_t frameBitsEnd = captured->bitsOffset + captured->bitsSize;

    if (count > SIZE_MAX - 255)
      return fail(plan, "lane length overflows", frameIndex, SIZE_MAX);
    const size_t stride = (count + 255) / 256 * 256;
    uint32_t laneBase, laneStride, frameCount32, outputBase;
    if (as_u32(plan->laneBytes, &laneBase) || as_u32(stride, &laneStride) ||
        as_u32(count, &frameCount32) || as_u32(shape->outOff, &outputBase))
      return fail(plan, "GPU offset exceeds 32 bits", frameIndex, SIZE_MAX);
    plan->joins[frameIndex] =
        (GpuJoinTask){laneBase, laneStride, frameCount32, outputBase};
    if (count > plan->maxFrameLength)
      plan->maxFrameLength = count;

    size_t weightsOffset = 0;
    size_t bitsOffset = captured->bitsOffset;
    for (size_t laneIndex = 0; laneIndex < elementBytes; ++laneIndex) {
      const uint8_t *laneHeader = header + PLANAR_ZIGZAG_PIVCO_HEADER_FIXED +
                                  laneIndex * PLANAR_ZIGZAG_PIVCO_HEADER_LANE;
      const size_t weightsSize = geozl_ld_le16(laneHeader + 1);
      const size_t bitsSize = geozl_ld_le32(laneHeader + 3);
      if (bitsOffset > frameBitsEnd || bitsSize > frameBitsEnd - bitsOffset ||
          weightsOffset > captured->weightsSize ||
          weightsSize > captured->weightsSize - weightsOffset)
        return fail(plan, "lane exceeds captured input", frameIndex, laneIndex);
      if (laneIndex > SIZE_MAX / stride ||
          plan->laneBytes > SIZE_MAX - laneIndex * stride)
        return fail(plan, "lane output offset overflows", frameIndex,
                    laneIndex);
      const size_t laneOutput = plan->laneBytes + laneIndex * stride;

      if (laneHeader[0] == PLANAR_ZIGZAG_PIVCO_LANE_PFOR) {
        if (weightsSize != 0)
          return fail(plan, "PFOR lane has Huffman weights", frameIndex,
                      laneIndex);
        ++plan->pfor_laneCount;
        const uint8_t *lanePayload = payload + bitsOffset;
        size_t blockOffset = 0;
        const size_t blocks = (count + GEOZL_PFOR_BLOCK - 1) / GEOZL_PFOR_BLOCK;
        if (RESERVE(plan, laneBlocks, laneCapacity, plan->laneBlockCount,
                    blocks + 4) != 0)
          return fail(plan, "out of memory", frameIndex, laneIndex);
        for (size_t blockIndex = 0; blockIndex < blocks; ++blockIndex) {
          PforBlockInfo block;
          if (pfor_block_info(lanePayload + blockOffset, bitsSize - blockOffset,
                              1, &block) != 0)
            return fail(plan, "invalid PFOR block", frameIndex, laneIndex);
          uint32_t payloadOffset32, outputOffset32;
          if (as_u32(bitsOffset + blockOffset, &payloadOffset32) ||
              as_u32(laneOutput + blockIndex * GEOZL_PFOR_BLOCK,
                     &outputOffset32))
            return fail(plan, "GPU offset exceeds 32 bits", frameIndex,
                        laneIndex);
          plan->laneBlocks[plan->laneBlockCount++] =
              (GpuLaneBlock){payloadOffset32, outputOffset32};
          blockOffset += block.bytes;
        }
        if (blockOffset != bitsSize)
          return fail(plan, "PFOR lane has trailing bytes", frameIndex,
                      laneIndex);
      } else if (laneHeader[0] == PLANAR_ZIGZAG_PIVCO_LANE_PIVCO) {
        if (weightsSize == 0)
          return fail(plan, "PivCo lane has no weights", frameIndex, laneIndex);
        ++plan->pivcoLaneCount;
        const uint8_t *weights = captured->weights + weightsOffset;
        const int tableLog =
            ZL_PivCoHuffman_computeTableLog(weights, weightsSize);
        if (tableLog < 0)
          return fail(plan, "invalid PivCo weights", frameIndex, laneIndex);
        ZL_PivCoHuffmanTree tree;
        ZL_PivCoHuffmanTree_build(&tree, weights, weightsSize, tableLog);
        topology.count = 0;
        if (topology_build(&topology, &tree, 0, 0, tree.numRanks) != 0)
          return fail(plan, "invalid PivCo topology", frameIndex, laneIndex);

        if (RESERVE(plan, symbols, symbolCapacity, plan->symbolCount,
                    ZL_PIVCO_MAX_SYMBOLS) ||
            RESERVE(plan, trees, treeCapacity, plan->treeCount, topology.count))
          return fail(plan, "out of memory", frameIndex, laneIndex);
        memcpy(plan->symbols + plan->symbolCount, tree.rankToSymbol,
               ZL_PIVCO_MAX_SYMBOLS);
        memcpy(plan->trees + plan->treeCount, topology.nodes,
               topology.count * sizeof(*plan->trees));

        BitCursor cursor = {payload + bitsOffset, 0, bitsSize * 8};
        for (size_t blockOffset = 0; blockOffset < count;
             blockOffset += blockSize) {
          const size_t blockLength =
              count - blockOffset < blockSize ? count - blockOffset : blockSize;
          if (RESERVE(plan, nodes, nodeCapacity, plan->nodeCount,
                      topology.count) ||
              RESERVE(plan, internalNodes, internalCapacity,
                      plan->internalNodeCount, topology.count + 4) ||
              RESERVE(plan, pivcoBlocks, pivcoCapacity, plan->pivcoBlockCount,
                      1))
            return fail(plan, "out of memory", frameIndex, laneIndex);
          if (walk_block(&topology, 0, blockLength, &cursor, bitsOffset,
                         plan->nodes + plan->nodeCount,
                         &plan->rankWordCount) != 0)
            return fail(plan, "invalid PivCo block", frameIndex, laneIndex);

          for (size_t i = 0; i < topology.count; ++i) {
            if (!(topology.nodes[i].leaf >> 16)) {
              uint32_t index;
              if (as_u32(plan->nodeCount + i, &index))
                return fail(plan, "GPU node index exceeds 32 bits", frameIndex,
                            laneIndex);
              plan->internalNodes[plan->internalNodeCount++] = index;
            }
          }

          GpuPivcoBlock gpuBlock;
          if (as_u32(plan->nodeCount, &gpuBlock.nodeBase) ||
              as_u32(plan->treeCount, &gpuBlock.treeBase) ||
              as_u32(plan->symbolCount, &gpuBlock.symbolBase) ||
              as_u32(laneOutput + blockOffset, &gpuBlock.outBase) ||
              as_u32(blockLength, &gpuBlock.length) ||
              as_u32(topology.count, &gpuBlock.nodeCount))
            return fail(plan, "GPU block field exceeds 32 bits", frameIndex,
                        laneIndex);
          plan->pivcoBlocks[plan->pivcoBlockCount++] = gpuBlock;
          plan->nodeCount += topology.count;
          if (blockLength > plan->maxPivcoBlockLength)
            plan->maxPivcoBlockLength = blockLength;
        }
        if ((cursor.bit + 7) / 8 != bitsSize)
          return fail(plan, "PivCo lane has trailing bytes", frameIndex,
                      laneIndex);
        plan->symbolCount += ZL_PIVCO_MAX_SYMBOLS;
        plan->treeCount += topology.count;
      } else {
        return fail(plan, "unknown lane mode", frameIndex, laneIndex);
      }

      weightsOffset += weightsSize;
      bitsOffset += bitsSize;
    }
    if (weightsOffset != captured->weightsSize || bitsOffset != frameBitsEnd)
      return fail(plan, "lane sizes do not cover captured input", frameIndex,
                  SIZE_MAX);
    if (elementBytes > SIZE_MAX / stride ||
        elementBytes * stride > SIZE_MAX - plan->laneBytes)
      return fail(plan, "lane buffer size overflows", frameIndex, SIZE_MAX);
    plan->laneBytes += elementBytes * stride;
  }

  // Four waves share a group. Rank tasks use a sentinel; PFOR padding repeats
  // a valid block into a spare sink run after the lane buffer.
  if (RESERVE(plan, internalNodes, internalCapacity, plan->internalNodeCount,
              4) ||
      RESERVE(plan, laneBlocks, laneCapacity, plan->laneBlockCount, 4))
    return fail(plan, "out of memory", frameCount - 1, SIZE_MAX);
  while (plan->internalNodeCount % 4)
    plan->internalNodes[plan->internalNodeCount++] = UINT32_MAX;
  while (plan->laneBlockCount % 4 && plan->laneBlockCount > 0) {
    uint32_t sink;
    if (as_u32(plan->laneBytes, &sink))
      return fail(plan, "lane sink exceeds 32 bits", frameCount - 1, SIZE_MAX);
    plan->laneBlocks[plan->laneBlockCount++] =
        (GpuLaneBlock){plan->laneBlocks[0].payloadOffset, sink};
  }

  // Metal and CUDA both require a non-empty bound buffer even when a kernel is
  // omitted because its task list is empty.
  if (RESERVE(plan, symbols, symbolCapacity, plan->symbolCount, 1) ||
      RESERVE(plan, trees, treeCapacity, plan->treeCount, 1) ||
      RESERVE(plan, nodes, nodeCapacity, plan->nodeCount, 1) ||
      RESERVE(plan, internalNodes, internalCapacity, plan->internalNodeCount,
              1) ||
      RESERVE(plan, pivcoBlocks, pivcoCapacity, plan->pivcoBlockCount, 1) ||
      RESERVE(plan, laneBlocks, laneCapacity, plan->laneBlockCount, 1))
    return fail(plan, "out of memory", frameCount - 1, SIZE_MAX);
  return 0;
}

#undef RESERVE
