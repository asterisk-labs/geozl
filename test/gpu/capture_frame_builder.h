#ifndef GEOZL_TEST_GPU_CAPTURE_FRAME_BUILDER_H
#define GEOZL_TEST_GPU_CAPTURE_FRAME_BUILDER_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
  uint32_t ctid;
  const uint8_t *header;
  size_t headerSize;
  size_t outputCount;
  unsigned numericOutputs;
  size_t outputBytes[2];
} CaptureFramePlan;

size_t capture_test_build_frame(uint8_t *dst, size_t capacity,
                                size_t outerBytes,
                                const CaptureFramePlan *plan);

#endif // GEOZL_TEST_GPU_CAPTURE_FRAME_BUILDER_H
