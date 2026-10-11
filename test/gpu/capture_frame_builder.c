#include "capture_frame_builder.h"

#include "openzl/codecs/zl_store.h"
#include "openzl/zl_compress.h"
#include "openzl/zl_compressor.h"
#include "openzl/zl_ctransform.h"
#include "openzl/zl_errors.h"
#include "openzl/zl_output.h"
#include "openzl/zl_version.h"

#include <string.h>

static ZL_Report encode_frame(ZL_Encoder *encoder, const ZL_Input *input) {
  (void)input;
  const CaptureFramePlan *plan = ZL_Encoder_getOpaquePtr(encoder);
  for (size_t i = 0; i < plan->outputCount; ++i) {
    ZL_Output *output =
        ZL_Encoder_createTypedStream(encoder, (int)i, plan->outputBytes[i], 1);
    if (output == NULL)
      return ZL_returnError(ZL_ErrorCode_allocation);
    if (plan->outputBytes[i] != 0)
      memset(ZL_Output_ptr(output), 0, plan->outputBytes[i]);
    const ZL_Report report = ZL_Output_commit(output, plan->outputBytes[i]);
    if (ZL_isError(report))
      return report;
  }
  ZL_Encoder_sendCodecHeader(encoder, plan->header, plan->headerSize);
  return ZL_returnValue(plan->outputCount);
}

size_t capture_test_build_frame(uint8_t *dst, size_t capacity,
                                size_t outerBytes,
                                const CaptureFramePlan *plan) {
  static const uint8_t seed[64] = {0};
  if (dst == NULL || plan == NULL || outerBytes == 0 ||
      outerBytes > sizeof(seed) || plan->outputCount == 0 ||
      plan->outputCount > 2 || (plan->header == NULL && plan->headerSize != 0))
    return 0;

  ZL_Type outputTypes[2];
  for (size_t i = 0; i < plan->outputCount; ++i)
    outputTypes[i] =
        (plan->numericOutputs & (1u << i)) ? ZL_Type_numeric : ZL_Type_serial;

  const ZL_TypedEncoderDesc desc = {
      .gd = {.CTid = plan->ctid,
             .inStreamType = ZL_Type_numeric,
             .outStreamTypes = outputTypes,
             .nbOutStreams = plan->outputCount},
      .transform_f = encode_frame,
      .name = "capture_test_frame",
      .opaque = {.ptr = (void *)plan},
  };

  size_t written = 0;
  ZL_Compressor *compressor = ZL_Compressor_create();
  if (compressor == NULL)
    return 0;
  const ZL_NodeID node = ZL_Compressor_registerTypedEncoder(compressor, &desc);
  if (!ZL_NodeID_isValid(node))
    goto done;

  const ZL_GraphID graph =
      plan->outputCount == 1
          ? ZL_Compressor_registerStaticGraph_fromNode(
                compressor, node, ZL_GRAPHLIST(ZL_GRAPH_STORE))
          : ZL_Compressor_registerStaticGraph_fromNode(
                compressor, node, ZL_GRAPHLIST(ZL_GRAPH_STORE, ZL_GRAPH_STORE));
  if (!ZL_GraphID_isValid(graph) ||
      ZL_isError(ZL_Compressor_selectStartingGraphID(compressor, graph)))
    goto done;

  ZL_CCtx *context = ZL_CCtx_create();
  if (context == NULL)
    goto done;
  if (!ZL_isError(ZL_CCtx_refCompressor(context, compressor)) &&
      !ZL_isError(ZL_CCtx_setParameter(context, ZL_CParam_formatVersion,
                                       ZL_MAX_FORMAT_VERSION))) {
    ZL_TypedRef *input = ZL_TypedRef_createNumeric(seed, 1, outerBytes);
    if (input != NULL) {
      const ZL_Report report =
          ZL_CCtx_compressTypedRef(context, dst, capacity, input);
      ZL_TypedRef_free(input);
      if (!ZL_isError(report))
        written = ZL_validResult(report);
    }
  }
  ZL_CCtx_free(context);

done:
  ZL_Compressor_free(compressor);
  return written;
}
