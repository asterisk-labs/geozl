#ifndef GEOZL_GPU_PROTO_PIVCO_PLAN_H
#define GEOZL_GPU_PROTO_PIVCO_PLAN_H

#include "bench.h"

#include "planar_zigzag_pivco/graph_planar_zigzag_pivco.h"

#include "openzl/codecs/pivco_huffman/common_pivco_kernel.h"

#include <stddef.h>
#include <stdint.h>

// What the codec's decoder received for one frame. The benchmark captures this
// after OpenZL has validated and decoded the frame on the CPU.
typedef struct {
  uint8_t header[PLANAR_ZIGZAG_PIVCO_HEADER_MAX];
  size_t headerSize;
  uint8_t weights[8 * ZL_PIVCO_MAX_SYMBOLS];
  size_t weightsSize;
  size_t bitsOffset;
  size_t bitsSize;
  int seen;
} CapturedFrame;

// These structures mirror the corresponding declarations in the Slang
// kernels. Keep every field 32-bit so the C and GPU layouts agree.
typedef struct {
  uint32_t links, leaf;
} GpuTreeNode;

typedef struct {
  uint32_t offset, count, ranks;
} GpuNodeBits;

typedef struct {
  uint32_t nodeBase, treeBase, symbolBase, outBase, length, nodeCount;
} GpuPivcoBlock;

typedef struct {
  uint32_t payloadOffset, outputBase;
} GpuLaneBlock;

typedef struct {
  uint32_t laneBase, laneStride, count, outputBase;
} GpuJoinTask;

typedef struct {
  uint8_t *symbols;
  GpuTreeNode *trees;
  GpuNodeBits *nodes;
  uint32_t *internalNodes;
  GpuPivcoBlock *pivcoBlocks;
  GpuLaneBlock *laneBlocks;
  GpuJoinTask *joins;

  size_t symbolCount;
  size_t treeCount;
  size_t nodeCount;
  size_t internalNodeCount;
  size_t pivcoBlockCount;
  size_t laneBlockCount;
  size_t rankWordCount;
  size_t laneBytes;
  size_t maxPivcoBlockLength;
  size_t maxFrameLength;
  size_t pivcoLaneCount;
  size_t pfor_laneCount;
} PivcoGpuPlan;

// Builds the immutable task lists consumed by planar_zigzag_pivco.slang.
// The captured frames must already have passed the CPU decoder.
int pivco_gpu_plan_build(PivcoGpuPlan *plan, const CapturedFrame *frames,
                         const Shape *shapes, size_t frameCount,
                         const uint8_t *payload, size_t payloadSize,
                         size_t elementBytes);

void pivco_gpu_plan_free(PivcoGpuPlan *plan);

#endif // GEOZL_GPU_PROTO_PIVCO_PLAN_H
