#ifndef GEOZL_GPU_PFOR_CAPTURE_H
#define GEOZL_GPU_PFOR_CAPTURE_H

#include "pfor/graph_pfor.h"
#include "planar_zigzag_pfor/graph_planar_zigzag_pfor.h"

#include "openzl/zl_decompress.h"

#include <stddef.h>
#include <stdint.h>

typedef struct {
  uint8_t header[PLANAR_ZIGZAG_PFOR_HEADER_SIZE];
  size_t headerSize;
  size_t payloadOffset;
  size_t payloadSize;
  uint32_t rawElementBytes;
  enum {
    PFOR_CAPTURE_KIND_NONE = 0,
    PFOR_CAPTURE_KIND_ID,
    PFOR_CAPTURE_KIND_PLANAR_ZIGZAG,
  } kind;
  int seen;
} CapturedPforFrame;

typedef enum {
  PFOR_CAPTURE_ERROR = -1,
  PFOR_CAPTURE_CODEC = 0,
  PFOR_CAPTURE_RAW = 1,
} PforCaptureResult;

typedef struct {
  ZL_DCtx *dctx;
  CapturedPforFrame *current;
  void *destination;
  size_t destinationCapacity;
  uint8_t *payload;
  size_t payloadSize;
  size_t payloadCapacity;
} PforCapture;

int pfor_capture_init(PforCapture *capture, uint8_t *payload,
                      size_t payloadCapacity);
void pfor_capture_destroy(PforCapture *capture);
PforCaptureResult pfor_capture_frame(PforCapture *capture,
                                     CapturedPforFrame *frame, void *dst,
                                     size_t dstCapacity, const void *src,
                                     size_t srcSize);

#endif // GEOZL_GPU_PFOR_CAPTURE_H
