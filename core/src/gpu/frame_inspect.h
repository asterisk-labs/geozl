#ifndef GEOZL_GPU_FRAME_INSPECT_H
#define GEOZL_GPU_FRAME_INSPECT_H

#include "openzl/decompress/decode_frameheader.h"

static inline int geozl_gpu_frame_is_raw(const void *src, size_t srcSize) {
  DFH_Struct frame = {0};
  DFH_init(&frame);
  const ZL_Report frameHeader = DFH_decodeFrameHeader(&frame, src, srcSize);
  if (ZL_isError(frameHeader)) {
    DFH_destroy(&frame);
    return 0;
  }
  const size_t frameHeaderSize = (size_t)ZL_validResult(frameHeader);
  if (frameHeaderSize > srcSize) {
    DFH_destroy(&frame);
    return 0;
  }
  const ZL_Report chunkHeader =
      DFH_decodeChunkHeader(&frame, (const uint8_t *)src + frameHeaderSize,
                            srcSize - frameHeaderSize);
  const size_t count = VECTOR_SIZE(frame.nodes);
  const int raw =
      !ZL_isError(chunkHeader) &&
      (count == 0 ||
       (count == 1 && VECTOR_AT(frame.nodes, 0).trpid.trt == trt_standard &&
        VECTOR_AT(frame.nodes, 0).trpid.trid ==
            ZL_StandardTransformID_convert_num_to_serial_le));
  DFH_destroy(&frame);
  return raw;
}

#endif // GEOZL_GPU_FRAME_INSPECT_H
