#include "geozl/geozl.h"

#include "average/encode_average_binding.h"
#include "binoffset/encode_binoffset_binding.h"
#include "deinterleave/encode_deinterleave_binding.h"
#include "delta_n/encode_delta_n_binding.h"
#include "delta_w/encode_delta_w_binding.h"
#include "floatmult/encode_floatmult_binding.h"
#include "floatquant/encode_floatquant_binding.h"
#include "intmult/encode_intmult_binding.h"
#include "med/encode_med_binding.h"
#include "planar/encode_planar_binding.h"
#include "quant_linear/encode_quant_linear_binding.h"
#include "wp_static/encode_wp_static_binding.h"

#include "common/graph_num1to1.h"            // GEOZL_PARAM_WIDTH
#include "quant_linear/graph_quant_linear.h" // QUANT_LINEAR_PARAM_*

#include "openzl/zl_compressor.h"
#include "openzl/zl_ctransform.h"
#include "openzl/zl_localParams.h"

#include <string.h>

// Register a width predictor and attach the row width, the one param it takes.
static ZL_NodeID width_node(ZL_Compressor *c, const ZL_TypedEncoderDesc *desc,
                            uint32_t width) {
  ZL_NodeID base = ZL_Compressor_registerTypedEncoder(c, desc);
  if (!ZL_NodeID_isValid(base))
    return base;
  ZL_LocalParams lp = ZL_LP_1INTPARAM(GEOZL_PARAM_WIDTH, (int)width);
  ZL_NodeParameters np = {.localParams = &lp};
  ZL_RESULT_OF(ZL_NodeID) r = ZL_Compressor_parameterizeNode(c, base, &np);
  return ZL_RES_isError(r) ? ZL_NODE_ILLEGAL : ZL_RES_value(r);
}

ZL_NodeID geozl_node_delta_w(ZL_Compressor *c, uint32_t width) {
  const ZL_TypedEncoderDesc desc = EI_DELTA_W(GEOZL_CTID_DELTA_W);
  return width_node(c, &desc, width);
}
ZL_NodeID geozl_node_delta_n(ZL_Compressor *c, uint32_t width) {
  const ZL_TypedEncoderDesc desc = EI_DELTA_N(GEOZL_CTID_DELTA_N);
  return width_node(c, &desc, width);
}
ZL_NodeID geozl_node_planar(ZL_Compressor *c, uint32_t width) {
  const ZL_TypedEncoderDesc desc = EI_PLANAR(GEOZL_CTID_PLANAR);
  return width_node(c, &desc, width);
}
ZL_NodeID geozl_node_med(ZL_Compressor *c, uint32_t width) {
  const ZL_TypedEncoderDesc desc = EI_MED(GEOZL_CTID_MED);
  return width_node(c, &desc, width);
}
ZL_NodeID geozl_node_average(ZL_Compressor *c, uint32_t width) {
  const ZL_TypedEncoderDesc desc = EI_AVERAGE(GEOZL_CTID_AVERAGE);
  return width_node(c, &desc, width);
}
ZL_NodeID geozl_node_wp_static(ZL_Compressor *c, uint32_t width) {
  const ZL_TypedEncoderDesc desc = EI_WP_STATIC(GEOZL_CTID_WP_STATIC);
  return width_node(c, &desc, width);
}

ZL_NodeID geozl_node_deinterleave(ZL_Compressor *c) {
  const ZL_TypedEncoderDesc desc = EI_DEINTERLEAVE(GEOZL_CTID_DEINTERLEAVE);
  return ZL_Compressor_registerTypedEncoder(c, &desc);
}

ZL_NodeID geozl_node_binoffset(ZL_Compressor *c) {
  const ZL_TypedEncoderDesc desc = EI_BINOFFSET(GEOZL_CTID_BINOFFSET);
  return ZL_Compressor_registerTypedEncoder(c, &desc);
}

ZL_NodeID geozl_node_intmult(ZL_Compressor *c, uint64_t base) {
  const ZL_TypedEncoderDesc desc = EI_INTMULT(GEOZL_CTID_INTMULT);
  ZL_NodeID node = ZL_Compressor_registerTypedEncoder(c, &desc);
  if (!ZL_NodeID_isValid(node))
    return node;
  ZL_LocalParams lp = ZL_LP_1INTPARAM(1 /* base */, (int)base);
  ZL_NodeParameters np = {.localParams = &lp};
  ZL_RESULT_OF(ZL_NodeID) r = ZL_Compressor_parameterizeNode(c, node, &np);
  return ZL_RES_isError(r) ? ZL_NODE_ILLEGAL : ZL_RES_value(r);
}

ZL_NodeID geozl_node_floatquant(ZL_Compressor *c, unsigned k) {
  const ZL_TypedEncoderDesc desc = EI_FLOATQUANT(GEOZL_CTID_FLOATQUANT);
  ZL_NodeID node = ZL_Compressor_registerTypedEncoder(c, &desc);
  if (!ZL_NodeID_isValid(node))
    return node;
  ZL_LocalParams lp = ZL_LP_1INTPARAM(1 /* k */, (int)k);
  ZL_NodeParameters np = {.localParams = &lp};
  ZL_RESULT_OF(ZL_NodeID) r = ZL_Compressor_parameterizeNode(c, node, &np);
  return ZL_RES_isError(r) ? ZL_NODE_ILLEGAL : ZL_RES_value(r);
}

ZL_NodeID geozl_node_floatmult(ZL_Compressor *c, double base) {
  const ZL_TypedEncoderDesc desc = EI_FLOATMULT(GEOZL_CTID_FLOATMULT);
  ZL_NodeID node = ZL_Compressor_registerTypedEncoder(c, &desc);
  if (!ZL_NodeID_isValid(node))
    return node;
  uint64_t bits;
  memcpy(&bits, &base, sizeof(double));
  ZL_IntParam ps[2] = {
      {.paramId = 1, .paramValue = (int)(uint32_t)(bits & 0xFFFFFFFFu)},
      {.paramId = 2, .paramValue = (int)(uint32_t)(bits >> 32)},
  };
  ZL_LocalParams lp = {.intParams = {.intParams = ps, .nbIntParams = 2}};
  ZL_NodeParameters np = {.localParams = &lp};
  ZL_RESULT_OF(ZL_NodeID) r = ZL_Compressor_parameterizeNode(c, node, &np);
  return ZL_RES_isError(r) ? ZL_NODE_ILLEGAL : ZL_RES_value(r);
}

ZL_NodeID geozl_node_quant_linear(ZL_Compressor *c, double max_error,
                                  int dtype) {
  // Integers store the reconstruction, signalled by a negative scale, so the
  // decoder only copies. Floats keep the index stream, their reconstruction is
  // a float and would not compress like the integer indices do.
  const double step = 2.0 * max_error;
  const double scale = dtype <= QL_I64 ? -step : step;
  const ZL_TypedEncoderDesc desc = EI_QUANT_LINEAR(GEOZL_CTID_QUANT_LINEAR);
  ZL_NodeID base = ZL_Compressor_registerTypedEncoder(c, &desc);
  if (!ZL_NodeID_isValid(base))
    return base;
  const ZL_IntParam ip = {.paramId = QUANT_LINEAR_PARAM_DTYPE,
                          .paramValue = dtype};
  const ZL_CopyParam cp = {.paramId = QUANT_LINEAR_PARAM_SCALE,
                           .paramPtr = &scale,
                           .paramSize = sizeof(scale)};
  ZL_LocalParams lp = {
      .intParams = {.intParams = &ip, .nbIntParams = 1},
      .copyParams = {.copyParams = &cp, .nbCopyParams = 1},
  };
  ZL_NodeParameters np = {.localParams = &lp};
  ZL_RESULT_OF(ZL_NodeID) r = ZL_Compressor_parameterizeNode(c, base, &np);
  return ZL_RES_isError(r) ? ZL_NODE_ILLEGAL : ZL_RES_value(r);
}
