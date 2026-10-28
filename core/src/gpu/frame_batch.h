#ifndef GEOZL_GPU_FRAME_BATCH_H
#define GEOZL_GPU_FRAME_BATCH_H

#include <stddef.h>
#include <stdint.h>

#define GEOZL_GPU_PAYLOAD_PADDING 64

// Mirrors raw_copy in the pfor and pivco programs. Counts are elements.
typedef struct {
  uint32_t source;
  uint32_t destination;
  uint32_t count;
} GeozlGpuRawCopy;

typedef struct {
  size_t totalBytes;
  size_t maxFrameBytes;
  size_t payloadCapacity; // Bytes the frame capture may fill.
  size_t payloadBytes;    // Allocation size, including the trailing guard.
} GeozlGpuFrameBatchInfo;

int geozl_gpu_frame_batch_inspect(const void *const frames[],
                                  const size_t frameSizes[], size_t frameCount,
                                  size_t elementBytes, size_t maxOutputBytes,
                                  size_t outputSizes[],
                                  GeozlGpuFrameBatchInfo *info, char *error,
                                  size_t errorCapacity);

#endif // GEOZL_GPU_FRAME_BATCH_H
