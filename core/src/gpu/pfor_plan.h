#ifndef GEOZL_GPU_PFOR_PLAN_H
#define GEOZL_GPU_PFOR_PLAN_H

#include "host_alloc.h"
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

// Must match block_task in pfor.slang.
typedef struct {
  uint32_t payloadOffset;
  uint32_t outputBase;
  uint32_t count;
  uint32_t pad;
} GpuPforBlock;

// These structs must match pfor_bands.slang.
typedef struct {
  uint32_t payloadOffset, payloadSize, blockCount, firstBlock, frame, pad[3];
} GpuPforBandFrame;

typedef struct {
  uint32_t header, rest;
} GpuPforBlockRef;

typedef struct {
  uint32_t frame, first, count, firstChunk;
} GpuPforBlockChunk;

typedef struct {
  uint32_t firstBlock, blockCount, outputBase, width, height, frame, columns,
      pad;
} GpuPforStrip;

typedef struct {
  GpuPforBandFrame *frames; // padded to whole groups of four with empty frames
  size_t frameCount;
  GpuPforStrip *strips; // padded the same way with empty strips
  size_t stripCount;
  GpuPforBlockChunk *chunks; // padded the same way with empty chunks
  size_t chunkCount;
  size_t blockCount; // GpuPforBlockRef entries the GPU writes
  int borrowed; // the arrays belong to the allocator the plan was built with
  const char *error;
  size_t errorFrame;
} PforGpuBandPlan;

int pfor_gpu_band_plan_build(PforGpuBandPlan *plan, const PforGpuFrame *frames,
                             size_t frameCount, const size_t frameIds[],
                             const GeozlGpuHostAlloc *alloc);
void pfor_gpu_band_plan_free(PforGpuBandPlan *plan);
int pfor_gpu_band_frame_fits(const PforGpuFrame *frame);

enum { PFOR_GPU_STATUS_HEADER = 0, PFOR_GPU_STATUS_POSITIONS = 1 };

typedef struct {
  GpuPforBlock *blocks;
  size_t blockCount;
  size_t realBlockCount;
  size_t blocksWithExceptions;
  size_t exceptionCount;
  int borrowed; // blocks belong to the allocator the plan was built with
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

typedef struct {
  size_t blocksWithExceptions;
  size_t exceptionCount;
  const char *error;
  size_t errorBlock;
} PforGpuFrameTally;

int pfor_gpu_plan_frame(GpuPforBlock blocks[], const PforGpuFrame *frame,
                        const uint8_t *payload, size_t elementBytes,
                        PforGpuFrameTally *tally);
// Wrap caller-owned blocks. Capacity must be realBlocks rounded up to four.
void pfor_gpu_plan_wrap(PforGpuPlan *plan, GpuPforBlock blocks[],
                        size_t realBlocks);

int pfor_gpu_plan_build(PforGpuPlan *plan, const PforGpuFrame *frames,
                        size_t frameCount, const uint8_t *payload,
                        size_t payloadSize, size_t elementBytes);
int pfor_gpu_plan_build_parallel(PforGpuPlan *plan, const PforGpuFrame *frames,
                                 size_t frameCount, const uint8_t *payload,
                                 size_t payloadSize, size_t elementBytes,
                                 size_t threadCount,
                                 const GeozlGpuHostAlloc *alloc);
void pfor_gpu_plan_free(PforGpuPlan *plan);

#endif // GEOZL_GPU_PFOR_PLAN_H
