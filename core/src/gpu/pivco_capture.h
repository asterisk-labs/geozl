#ifndef GEOZL_GPU_PIVCO_CAPTURE_H
#define GEOZL_GPU_PIVCO_CAPTURE_H

#include "planar_zigzag_transpose_pivco/graph_planar_zigzag_transpose_pivco.h"
#include "transpose_pivco/graph_transpose_pivco.h"

#include "openzl/codecs/pivco_huffman/common_pivco_kernel.h"
#include "openzl/zl_decompress.h"

#include <stddef.h>
#include <stdint.h>

typedef struct {
  uint8_t header[PLANAR_ZIGZAG_TRANSPOSE_PIVCO_HEADER_MAX];
  size_t headerSize;
  uint8_t weights[8 * ZL_PIVCO_MAX_SYMBOLS];
  size_t weightsSize;
  size_t bitsOffset;
  size_t bitsSize;
  uint32_t rawElementBytes;
  enum {
    PIVCO_CAPTURE_KIND_NONE = 0,
    PIVCO_CAPTURE_KIND_ID,
    PIVCO_CAPTURE_KIND_PLANAR_ZIGZAG,
  } kind;
  // 0 before the codec, 1 at the root, 2 when another host transform follows,
  // and -1 for invalid codec metadata.
  int seen;
} CapturedFrame;

typedef enum {
  PIVCO_CAPTURE_ERROR = -1,
  PIVCO_CAPTURE_CODEC = 0,
  PIVCO_CAPTURE_RAW = 1,
  PIVCO_CAPTURE_FALLBACK = 2,
} PivcoCaptureResult;

typedef struct {
  ZL_DCtx *dctx;
  CapturedFrame *current;
  void *destination;
  size_t destinationCapacity;
  uint8_t *payload;
  size_t payloadSize;
  size_t payloadCapacity;
} PivcoCapture;

int pivco_capture_init(PivcoCapture *capture, uint8_t *payload,
                       size_t payloadCapacity);
void pivco_capture_destroy(PivcoCapture *capture);

PivcoCaptureResult pivco_capture_frame(PivcoCapture *capture,
                                       CapturedFrame *frame, void *dst,
                                       size_t dstCapacity, const void *src,
                                       size_t srcSize);

#endif // GEOZL_GPU_PIVCO_CAPTURE_H
