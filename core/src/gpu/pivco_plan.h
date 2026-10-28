#ifndef GEOZL_GPU_PIVCO_PLAN_H
#define GEOZL_GPU_PIVCO_PLAN_H

#include "host_alloc.h"
#include "pivco_capture.h"
#include "planar_scan.h"

#include <stddef.h>
#include <stdint.h>

// These structures mirror the corresponding declarations in the Slang
// kernels. Keep every field 32-bit so the C and GPU layouts agree.
typedef struct {
  uint32_t links, leaf;
} GpuTreeNode;

typedef struct {
  uint32_t offset, count, output;
} GpuNodeBits;

typedef struct {
  uint32_t nodeBase, treeBase, symbolBase, outBase, length, nodeCount;
} GpuPivcoBlock;

typedef struct {
  uint32_t payloadOffset, outputBase;
} GpuLaneBlock;

typedef struct {
  uint32_t laneBase, laneStride, count, outputBase, start, transform;
} GpuJoinTask;

typedef struct {
  uint8_t *symbols;
  GpuTreeNode *trees;
  GpuNodeBits *nodes;
  GpuPivcoBlock *pivcoBlocks;
  GpuLaneBlock *laneBlocks;
  GpuJoinTask *joins;

  size_t symbolCount;
  size_t treeCount;
  size_t nodeCount;
  size_t bitmapCount;
  size_t pivcoBlockCount;
  size_t laneBlockCount;
  size_t joinCount;
  size_t laneBytes;
  size_t pivcoLaneCount;
  size_t pforLaneCount;
  int borrowed; // the arrays belong to the allocator the plan was built with

  const char *error;
  size_t errorFrame;
  size_t errorLane;
} PivcoGpuPlan;

// Smaller blocks would let a tiny payload plan far more nodes than it decodes.
// GeoZL writes 32768-symbol blocks.
#define GEOZL_GPU_PIVCO_MIN_BLOCK 1024

// Builds the immutable task lists consumed by gpu/kernels/programs/pivco.slang.
// The capture step validates the outer frame, but this planner is responsible
// for validating the PivCo and PFOR streams that the CPU decoder did not run.
int pivco_gpu_plan_build(PivcoGpuPlan *plan, const CapturedFrame *frames,
                         const GeozlGpuFrameShape *shapes, size_t frameCount,
                         const uint8_t *payload, size_t payloadSize,
                         size_t elementBytes);

// Plans contiguous frame ranges on up to threadCount threads, then merges them
// in input order into arrays from alloc.
int pivco_gpu_plan_build_parallel(PivcoGpuPlan *plan,
                                  const CapturedFrame *frames,
                                  const GeozlGpuFrameShape *shapes,
                                  size_t frameCount, const uint8_t *payload,
                                  size_t payloadSize, size_t elementBytes,
                                  size_t threadCount,
                                  const GeozlGpuHostAlloc *alloc);

void pivco_gpu_plan_free(PivcoGpuPlan *plan);

#endif // GEOZL_GPU_PIVCO_PLAN_H
