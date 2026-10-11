#include "pfor_capture.h"

#include "frame_inspect.h"

#include "geozl/ctids.h"

#include "pfor/decode_pfor_binding.h"
#include "planar_zigzag_pfor/decode_planar_zigzag_pfor_binding.h"

#include "common/endian.h"
#include "pfor/pfor_check.h"

#include "openzl/zl_errors.h"
#include "openzl/zl_input.h"
#include "openzl/zl_output.h"

#include <string.h>

static ZL_Report capture_decoder(ZL_Decoder *decoder,
                                 const ZL_Input *inputs[]) {
  PforCapture *capture = (PforCapture *)ZL_Decoder_getOpaquePtr(decoder);
  CapturedPforFrame *frame = capture == NULL ? NULL : capture->current;
  const ZL_RBuffer header = ZL_Decoder_getCodecHeader(decoder);

  if (frame == NULL || frame->seen != 0 || inputs == NULL ||
      inputs[0] == NULL || header.start == NULL ||
      header.size < PFOR_HEADER_SIZE ||
      ZL_Input_type(inputs[0]) != ZL_Type_serial ||
      ZL_Input_eltWidth(inputs[0]) != 1) {
    if (frame != NULL)
      frame->seen = -1;
    return ZL_returnError(ZL_ErrorCode_corruption);
  }

  const uint8_t *bytes = header.start;
  const uint64_t count = geozl_ld_le64(bytes);
  const size_t elementBytes = bytes[8];
  frame->kind = header.size == PFOR_HEADER_SIZE ? PFOR_CAPTURE_KIND_ID
                : header.size == PLANAR_ZIGZAG_PFOR_HEADER_SIZE
                    ? PFOR_CAPTURE_KIND_PLANAR_ZIGZAG
                    : PFOR_CAPTURE_KIND_NONE;
  if (frame->kind == PFOR_CAPTURE_KIND_NONE || count == 0 ||
      !geozl_pfor_width_ok(elementBytes) || count > SIZE_MAX / elementBytes) {
    frame->seen = -1;
    return ZL_returnError(ZL_ErrorCode_corruption);
  }
  // The outer frame already fixed the caller-owned output size. Check the
  // codec header before asking OpenZL to create its output stream.
  if ((size_t)count * elementBytes != capture->destinationCapacity) {
    frame->seen = -1;
    return ZL_returnError(ZL_ErrorCode_corruption);
  }
  if (frame->kind == PFOR_CAPTURE_KIND_PLANAR_ZIGZAG) {
    const uint32_t width = geozl_ld_le32(bytes + 9);
    const uint32_t planes = geozl_ld_le32(bytes + 13);
    if (planes == 0 || width == 0 || count % planes != 0 ||
        width > count / planes || count / planes % width != 0) {
      frame->seen = -1;
      return ZL_returnError(ZL_ErrorCode_corruption);
    }
  }

  const size_t payloadSize = ZL_Input_contentSize(inputs[0]);
  const size_t paddedSize = payloadSize <= SIZE_MAX - 15
                                ? (payloadSize + 15) & ~(size_t)15
                                : SIZE_MAX;
  const uint8_t *payload = ZL_Input_ptr(inputs[0]);
  if (header.size > sizeof(frame->header) ||
      (payloadSize != 0 && payload == NULL) ||
      capture->payloadSize > capture->payloadCapacity ||
      paddedSize > capture->payloadCapacity - capture->payloadSize) {
    frame->seen = -1;
    return ZL_returnError(ZL_ErrorCode_allocation);
  }

  memcpy(frame->header, header.start, header.size);
  if (payloadSize != 0)
    memcpy(capture->payload + capture->payloadSize, payload, payloadSize);
  if (paddedSize != payloadSize)
    memset(capture->payload + capture->payloadSize + payloadSize, 0,
           paddedSize - payloadSize);
  frame->headerSize = header.size;
  frame->payloadOffset = capture->payloadSize;
  frame->payloadSize = payloadSize;
  frame->seen = 1;
  capture->payloadSize += paddedSize;
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

int pfor_capture_init(PforCapture *capture, uint8_t *payload,
                      size_t payloadCapacity) {
  if (capture == NULL || (payload == NULL && payloadCapacity != 0))
    return 1;
  *capture = (PforCapture){
      .dctx = ZL_DCtx_create(),
      .payload = payload,
      .payloadCapacity = payloadCapacity,
  };
  if (capture->dctx == NULL)
    return 1;

  ZL_TypedDecoderDesc desc =
      DI_PLANAR_ZIGZAG_PFOR(GEOZL_CTID_PLANAR_ZIGZAG_PFOR);
  desc.transform_f = capture_decoder;
  desc.opaque.ptr = capture;
  ZL_Report report = ZL_DCtx_registerTypedDecoder(capture->dctx, &desc);
  if (!ZL_isError(report)) {
    desc = (ZL_TypedDecoderDesc)DI_PFOR(GEOZL_CTID_PFOR);
    desc.transform_f = capture_decoder;
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
    pfor_capture_destroy(capture);
    return 1;
  }
  return 0;
}

void pfor_capture_destroy(PforCapture *capture) {
  if (capture == NULL)
    return;
  ZL_DCtx_free(capture->dctx);
  *capture = (PforCapture){0};
}

PforCaptureResult pfor_capture_frame(PforCapture *capture,
                                     CapturedPforFrame *frame, void *dst,
                                     size_t dstCapacity, const void *src,
                                     size_t srcSize) {
  if (capture == NULL || capture->dctx == NULL || frame == NULL ||
      dst == NULL || src == NULL)
    return PFOR_CAPTURE_ERROR;

  *frame = (CapturedPforFrame){0};
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
    return PFOR_CAPTURE_CODEC;
  if (frame->seen == 0 && !ZL_isError(report) &&
      geozl_gpu_frame_is_raw(src, srcSize) &&
      outputInfo.type == ZL_Type_numeric &&
      outputInfo.decompressedByteSize == dstCapacity) {
    frame->rawElementBytes = outputInfo.fixedWidth;
    return PFOR_CAPTURE_RAW;
  }
  return PFOR_CAPTURE_ERROR;
}
