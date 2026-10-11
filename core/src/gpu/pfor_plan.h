#ifndef GEOZL_GPU_PFOR_PLAN_H
#define GEOZL_GPU_PFOR_PLAN_H

#include "planar_scan.h"

#include <stddef.h>
#include <stdint.h>

typedef struct {
  GeozlGpuFrameShape shape;
  size_t payloadOffset;
  size_t payloadSize;
  enum {
    PFOR_GPU_ID = 0,
    PFOR_GPU_PLANAR_ZIGZAG = 1,
  } transform;
} PforGpuFrame;

// Mirrors block_task in gpu/kernels/stages/pfor.slang.
typedef struct {
  uint32_t payloadOffset;
  uint32_t outputBase;
  uint32_t count;
  uint32_t transform;
} GpuPforBlock;

typedef struct {
  GpuPforBlock *blocks;
  size_t blockCount;
  size_t realBlockCount;
  size_t blocksWithExceptions;
  size_t exceptionCount;
  const char *error;
  size_t errorFrame;
  size_t errorBlock;
} PforGpuPlan;

typedef struct {
  size_t bytes;
  unsigned exceptions;
} PforGpuBlockInfo;

int pfor_gpu_block_info(const uint8_t *block, size_t available,
                        size_t elementBytes, PforGpuBlockInfo *info);

int pfor_gpu_plan_build(PforGpuPlan *plan, const PforGpuFrame *frames,
                        size_t frameCount, const uint8_t *payload,
                        size_t payloadSize, size_t elementBytes);
int pfor_gpu_plan_build_parallel(PforGpuPlan *plan, const PforGpuFrame *frames,
                                 size_t frameCount, const uint8_t *payload,
                                 size_t payloadSize, size_t elementBytes,
                                 size_t threadCount);
void pfor_gpu_plan_free(PforGpuPlan *plan);

#endif // GEOZL_GPU_PFOR_PLAN_H
