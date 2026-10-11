#include "frame_batch.h"

#include "runtime_internal.h"

#include "openzl/zl_decompress.h"
#include "openzl/zl_errors.h"

#include <limits.h>
#include <stdint.h>

int geozl_gpu_frame_batch_inspect(const void *const frames[],
                                  const size_t frameSizes[], size_t frameCount,
                                  size_t elementBytes, size_t maxOutputBytes,
                                  size_t outputSizes[],
                                  GeozlGpuFrameBatchInfo *info, char *error,
                                  size_t errorCapacity) {
  if (frames == NULL || frameSizes == NULL || outputSizes == NULL ||
      info == NULL || frameCount == 0 || frameCount > UINT32_MAX ||
      (elementBytes != 1 && elementBytes != 2))
    return geozl_gpu_error(error, errorCapacity, "invalid GPU frame batch");

  *info = (GeozlGpuFrameBatchInfo){0};
  for (size_t i = 0; i < frameCount; ++i) {
    if (frames[i] == NULL || frameSizes[i] == 0)
      return geozl_gpu_error(error, errorCapacity, "invalid frame %zu", i);
    const ZL_Report report = ZL_getDecompressedSize(frames[i], frameSizes[i]);
    if (ZL_isError(report) || ZL_validResult(report) == 0)
      return geozl_gpu_error(error, errorCapacity,
                             "cannot read the size of frame %zu", i);
    outputSizes[i] = (size_t)ZL_validResult(report);
    if (outputSizes[i] % elementBytes != 0 ||
        outputSizes[i] > SIZE_MAX - info->totalBytes)
      return geozl_gpu_error(error, errorCapacity,
                             "output size overflows at frame %zu", i);
    info->totalBytes += outputSizes[i];
    if ((maxOutputBytes != 0 && info->totalBytes > maxOutputBytes) ||
        info->totalBytes > UINT32_MAX)
      return geozl_gpu_error(error, errorCapacity,
                             "GPU output is too large at frame %zu", i);
    if (outputSizes[i] > info->maxFrameBytes)
      info->maxFrameBytes = outputSizes[i];

    if (frameSizes[i] > SIZE_MAX - outputSizes[i] ||
        frameSizes[i] + outputSizes[i] > SIZE_MAX - 15 ||
        info->payloadCapacity >
            SIZE_MAX - (frameSizes[i] + outputSizes[i] + (size_t)15))
      return geozl_gpu_error(error, errorCapacity,
                             "payload size overflows at frame %zu", i);
    info->payloadCapacity += frameSizes[i] + outputSizes[i] + 15;
  }
  if (info->payloadCapacity > SIZE_MAX - GEOZL_GPU_PAYLOAD_PADDING)
    return geozl_gpu_error(error, errorCapacity, "payload capacity overflows");
  info->payloadBytes = info->payloadCapacity + GEOZL_GPU_PAYLOAD_PADDING;
  return 0;
}
