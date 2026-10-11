#include "pivco_capture.h"

#include "frame_inspect.h"

#include "geozl/ctids.h"

#include "planar_zigzag_transpose_pivco/decode_planar_zigzag_transpose_pivco_binding.h"
#include "transpose_pivco/decode_transpose_pivco_binding.h"

#include "common/endian.h"
#include "pivco_lanes/pivco_lanes.h"

#include "openzl/zl_errors.h"
#include "openzl/zl_input.h"
#include "openzl/zl_output.h"

#include <string.h>

static ZL_Report capture_decoder(ZL_Decoder *decoder, const ZL_Input *inputs[],
                                 int kind) {
  PivcoCapture *capture = (PivcoCapture *)ZL_Decoder_getOpaquePtr(decoder);
  CapturedFrame *frame = capture == NULL ? NULL : capture->current;
  const ZL_RBuffer header = ZL_Decoder_getCodecHeader(decoder);

  if (frame == NULL || frame->seen != 0 || inputs == NULL ||
      inputs[0] == NULL || inputs[1] == NULL || header.start == NULL ||
      header.size < 9 || ZL_Input_type(inputs[0]) != ZL_Type_numeric ||
      ZL_Input_eltWidth(inputs[0]) != 1 ||
      ZL_Input_type(inputs[1]) != ZL_Type_serial ||
      ZL_Input_eltWidth(inputs[1]) != 1) {
    if (frame != NULL)
      frame->seen = -1;
    return ZL_returnError(ZL_ErrorCode_corruption);
  }

  const uint8_t *bytes = header.start;
  const size_t elementBytes = bytes[8];
  if (!geozl_pivco_lanes_width_ok(elementBytes)) {
    frame->seen = -1;
    return ZL_returnError(ZL_ErrorCode_corruption);
  }
  const size_t fixedHeader = kind == PIVCO_CAPTURE_KIND_PLANAR_ZIGZAG
                                 ? PLANAR_ZIGZAG_TRANSPOSE_PIVCO_HEADER_FIXED
                                 : TRANSPOSE_PIVCO_HEADER_FIXED;
  if ((kind != PIVCO_CAPTURE_KIND_ID &&
       kind != PIVCO_CAPTURE_KIND_PLANAR_ZIGZAG) ||
      header.size !=
          fixedHeader + elementBytes * GEOZL_PIVCO_LANE_HEADER_SIZE) {
    frame->seen = -1;
    return ZL_returnError(ZL_ErrorCode_corruption);
  }
  frame->kind = kind;

  const uint64_t count = geozl_ld_le64(bytes);
  if (count == 0 || count > SIZE_MAX / elementBytes) {
    frame->seen = -1;
    return ZL_returnError(ZL_ErrorCode_corruption);
  }
  // The outer frame already fixed the caller-owned output size. Check the
  // codec header before asking OpenZL to create its output stream.
  if ((size_t)count * elementBytes != capture->destinationCapacity) {
    frame->seen = -1;
    return ZL_returnError(ZL_ErrorCode_corruption);
  }
  if (frame->kind == PIVCO_CAPTURE_KIND_PLANAR_ZIGZAG) {
    const uint32_t width = geozl_ld_le32(bytes + 9);
    const uint32_t planes = geozl_ld_le32(bytes + 13);
    if (width == 0 || planes == 0 || count % planes != 0 ||
        width > count / planes || count / planes % width != 0) {
      frame->seen = -1;
      return ZL_returnError(ZL_ErrorCode_corruption);
    }
  }

  const size_t weightsSize = ZL_Input_contentSize(inputs[0]);
  const size_t bitsSize = ZL_Input_contentSize(inputs[1]);
  const size_t paddedBitsSize =
      bitsSize <= SIZE_MAX - 15 ? (bitsSize + 15) & ~(size_t)15 : SIZE_MAX;
  const uint8_t *weights = ZL_Input_ptr(inputs[0]);
  const uint8_t *bits = ZL_Input_ptr(inputs[1]);
  if (header.size > sizeof(frame->header) ||
      weightsSize > sizeof(frame->weights) ||
      (weightsSize != 0 && weights == NULL) ||
      (bitsSize != 0 && bits == NULL) ||
      capture->payloadSize > capture->payloadCapacity ||
      paddedBitsSize > capture->payloadCapacity - capture->payloadSize) {
    frame->seen = -1;
    return ZL_returnError(ZL_ErrorCode_allocation);
  }

  memcpy(frame->header, bytes, header.size);
  if (weightsSize != 0)
    memcpy(frame->weights, weights, weightsSize);
  if (bitsSize != 0)
    memcpy(capture->payload + capture->payloadSize, bits, bitsSize);
  if (paddedBitsSize != bitsSize)
    memset(capture->payload + capture->payloadSize + bitsSize, 0,
           paddedBitsSize - bitsSize);
  frame->headerSize = header.size;
  frame->weightsSize = weightsSize;
  frame->bitsOffset = capture->payloadSize;
  frame->bitsSize = bitsSize;
  frame->seen = 1;
  capture->payloadSize += paddedBitsSize;

  ZL_Output *output =
      ZL_Decoder_create1OutStream(decoder, (size_t)count, elementBytes);
  if (output == NULL) {
    frame->seen = -1;
    return ZL_returnError(ZL_ErrorCode_allocation);
  }
  // A root codec writes into the caller's destination. A different pointer
  // means another codec still has to consume this output on the CPU.
  if (ZL_Output_ptr(output) != capture->destination) {
    frame->seen = -1;
    return ZL_returnError(ZL_ErrorCode_graph_invalid);
  }
  return ZL_returnError(ZL_ErrorCode_temporaryLibraryLimitation);
}

static ZL_Report capture_planar_decoder(ZL_Decoder *decoder,
                                        const ZL_Input *inputs[]) {
  return capture_decoder(decoder, inputs, PIVCO_CAPTURE_KIND_PLANAR_ZIGZAG);
}

static ZL_Report capture_transpose_decoder(ZL_Decoder *decoder,
                                           const ZL_Input *inputs[]) {
  return capture_decoder(decoder, inputs, PIVCO_CAPTURE_KIND_ID);
}

int pivco_capture_init(PivcoCapture *capture, uint8_t *payload,
                       size_t payloadCapacity) {
  if (capture == NULL || (payload == NULL && payloadCapacity != 0))
    return 1;

  *capture = (PivcoCapture){
      .dctx = ZL_DCtx_create(),
      .payload = payload,
      .payloadCapacity = payloadCapacity,
  };
  if (capture->dctx == NULL)
    return 1;

  ZL_TypedDecoderDesc desc = DI_PLANAR_ZIGZAG_TRANSPOSE_PIVCO(
      GEOZL_CTID_PLANAR_ZIGZAG_TRANSPOSE_PIVCO);
  desc.transform_f = capture_planar_decoder;
  desc.opaque.ptr = capture;
  ZL_Report report = ZL_DCtx_registerTypedDecoder(capture->dctx, &desc);
  if (!ZL_isError(report)) {
    desc = (ZL_TypedDecoderDesc)DI_TRANSPOSE_PIVCO(GEOZL_CTID_TRANSPOSE_PIVCO);
    desc.transform_f = capture_transpose_decoder;
    desc.opaque.ptr = capture;
    report = ZL_DCtx_registerTypedDecoder(capture->dctx, &desc);
  }
  if (!ZL_isError(report))
    report = ZL_DCtx_setParameter(capture->dctx, ZL_DParam_stickyParameters, 1);
  if (!ZL_isError(report))
    report =
        ZL_DCtx_setParameter(capture->dctx, ZL_DParam_checkCompressedChecksum,
                             ZL_TernaryParam_enable);
  if (ZL_isError(report)) {
    pivco_capture_destroy(capture);
    return 1;
  }
  return 0;
}

void pivco_capture_destroy(PivcoCapture *capture) {
  if (capture == NULL)
    return;
  ZL_DCtx_free(capture->dctx);
  *capture = (PivcoCapture){0};
}

PivcoCaptureResult pivco_capture_frame(PivcoCapture *capture,
                                       CapturedFrame *frame, void *dst,
                                       size_t dstCapacity, const void *src,
                                       size_t srcSize) {
  if (capture == NULL || capture->dctx == NULL || frame == NULL ||
      dst == NULL || src == NULL)
    return PIVCO_CAPTURE_ERROR;

  *frame = (CapturedFrame){0};
  capture->current = frame;
  capture->destination = dst;
  capture->destinationCapacity = dstCapacity;
  ZL_OutputInfo outputInfo;
  const ZL_Report report = ZL_DCtx_decompressTyped(
      capture->dctx, &outputInfo, dst, dstCapacity, src, srcSize);
  capture->current = NULL;
  capture->destination = NULL;
  capture->destinationCapacity = 0;
  if (frame->seen == 1 && ZL_isError(report) &&
      ZL_errorCode(report) == ZL_ErrorCode_temporaryLibraryLimitation)
    return PIVCO_CAPTURE_CODEC;
  if (frame->seen == 0 && !ZL_isError(report) &&
      geozl_gpu_frame_is_raw(src, srcSize) &&
      outputInfo.type == ZL_Type_numeric &&
      outputInfo.decompressedByteSize == dstCapacity) {
    frame->rawElementBytes = outputInfo.fixedWidth;
    return PIVCO_CAPTURE_RAW;
  }
  return PIVCO_CAPTURE_ERROR;
}
