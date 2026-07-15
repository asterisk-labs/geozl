// geozl 2d high-level API: predictor selector and tile compression.

#include "geozl/geozl.h"

#include "openzl/zl_compress.h"
#include "openzl/zl_compressor.h"
#include "openzl/zl_errors_types.h"
#include "openzl/codecs/zl_brute_force_selector.h"
#include "openzl/codecs/zl_delta.h"   // ZL_NODE_DELTA_INT
#include "openzl/codecs/zl_entropy.h" // ZL_GRAPH_ENTROPY
#include "openzl/codecs/zl_illegal.h" // ZL_GRAPH_ILLEGAL
#include "openzl/codecs/zl_zigzag.h"  // ZL_NODE_ZIGZAG

#include <string.h>

typedef enum {
  GEOZL_PRED_DELTA_W = 0,
  GEOZL_PRED_DELTA_N,
  GEOZL_PRED_PLANAR,
  GEOZL_PRED_MED,
  GEOZL_PRED_AVERAGE,
  GEOZL_PRED_WP_STATIC,
  GEOZL_PRED_DELTA_1D,
  GEOZL_PRED_COUNT
} geozl_predictor;

static ZL_NodeID predictor_node(ZL_Compressor *c, geozl_predictor p,
                                uint32_t width) {
  switch (p) {
  case GEOZL_PRED_DELTA_W:
    return geozl_node_delta_w(c, width);
  case GEOZL_PRED_DELTA_N:
    return geozl_node_delta_n(c, width);
  case GEOZL_PRED_PLANAR:
    return geozl_node_planar(c, width);
  case GEOZL_PRED_MED:
    return geozl_node_med(c, width);
  case GEOZL_PRED_AVERAGE:
    return geozl_node_average(c, width);
  case GEOZL_PRED_WP_STATIC:
    return geozl_node_wp_static(c, width);
  default:
    return ZL_NODE_ILLEGAL;
  }
}

static ZL_GraphID predictor_graph(ZL_Compressor *c, geozl_predictor p,
                                  uint32_t width) {
  if (p == GEOZL_PRED_DELTA_1D)
    return ZL_Compressor_registerStaticGraph_fromNode1o(c, ZL_NODE_DELTA_INT,
                                                        ZL_GRAPH_ENTROPY);
  ZL_NodeID pred = predictor_node(c, p, width);
  if (!ZL_NodeID_isValid(pred))
    return ZL_GRAPH_ILLEGAL;
  const ZL_NodeID pipeline[2] = {pred, ZL_NODE_ZIGZAG};
  return ZL_Compressor_registerStaticGraph_fromPipelineNodes1o(
      c, pipeline, 2, ZL_GRAPH_ENTROPY);
}

static ZL_GraphID geozl_2d_graph(ZL_Compressor *c, const geozl_predictor *predictors,
                          size_t nbPredictors, uint32_t width, double max_error,
                          int dtype) {
  if (predictors == NULL || nbPredictors == 0 ||
      nbPredictors > GEOZL_PRED_COUNT)
    return ZL_GRAPH_ILLEGAL;

  ZL_GraphID succ[GEOZL_PRED_COUNT];
  for (size_t i = 0; i < nbPredictors; ++i) {
    succ[i] = predictor_graph(c, predictors[i], width);
    if (!ZL_GraphID_isValid(succ[i]))
      return ZL_GRAPH_ILLEGAL;
  }

  ZL_GraphID sel =
      ZL_Compressor_registerBruteForceSelectorGraph(c, succ, nbPredictors);
  if (!ZL_GraphID_isValid(sel) || max_error < 0.0)
    return sel;

  ZL_NodeID q = geozl_node_quant_linear(c, max_error, dtype);
  if (!ZL_NodeID_isValid(q))
    return ZL_GRAPH_ILLEGAL;
  return ZL_Compressor_registerStaticGraph_fromNode1o(c, q, sel);
}

static int geozl_2d_method(const char *name, geozl_predictor *out,
                           size_t *outN) {
  if (name == NULL || out == NULL || outN == NULL)
    return -1;

  static const geozl_predictor full[] = {
      GEOZL_PRED_DELTA_W, GEOZL_PRED_DELTA_N,   GEOZL_PRED_PLANAR,
      GEOZL_PRED_MED,     GEOZL_PRED_AVERAGE,   GEOZL_PRED_WP_STATIC,
      GEOZL_PRED_DELTA_1D};
  if (strcmp(name, "full") == 0) {
    for (size_t i = 0; i < sizeof(full) / sizeof(full[0]); ++i)
      out[i] = full[i];
    *outN = sizeof(full) / sizeof(full[0]);
    return 0;
  }

  static const struct {
    const char *name;
    geozl_predictor id;
  } one[] = {
      {"delta_w", GEOZL_PRED_DELTA_W}, {"delta_n", GEOZL_PRED_DELTA_N},
      {"planar", GEOZL_PRED_PLANAR},   {"med", GEOZL_PRED_MED},
      {"average", GEOZL_PRED_AVERAGE}, {"wp_static", GEOZL_PRED_WP_STATIC},
      {"delta_1d", GEOZL_PRED_DELTA_1D},
  };
  for (size_t i = 0; i < sizeof(one) / sizeof(one[0]); ++i) {
    if (strcmp(name, one[i].name) == 0) {
      *out = one[i].id;
      *outN = 1;
      return 0;
    }
  }
  return -1;
}

ZL_Report geozl_2d_compress(const char *method, uint32_t width,
                            double max_error, int dtype, int formatVersion,
                            const void *src, size_t numElts, size_t eltWidth,
                            void *dst, size_t dstCapacity, size_t *outSize) {
  geozl_predictor set[GEOZL_PRED_COUNT];
  size_t n = 0;
  if (geozl_2d_method(method, set, &n) != 0)
    return ZL_returnError(ZL_ErrorCode_graph_invalid);

  ZL_Compressor *c = ZL_Compressor_create();
  if (c == NULL)
    return ZL_returnError(ZL_ErrorCode_allocation);

  ZL_Report r;
  ZL_TypedRef *in = NULL;
  ZL_CCtx *cctx = NULL;

  ZL_GraphID g = geozl_2d_graph(c, set, n, width, max_error, dtype);
  if (!ZL_GraphID_isValid(g)) {
    r = ZL_returnError(ZL_ErrorCode_graph_invalid);
    goto done;
  }
  r = ZL_Compressor_selectStartingGraphID(c, g);
  if (ZL_isError(r))
    goto done;

  cctx = ZL_CCtx_create();
  if (cctx == NULL) {
    r = ZL_returnError(ZL_ErrorCode_allocation);
    goto done;
  }
  r = ZL_CCtx_refCompressor(cctx, c);
  if (ZL_isError(r))
    goto done;
  r = ZL_CCtx_setParameter(cctx, ZL_CParam_formatVersion, formatVersion);
  if (ZL_isError(r))
    goto done;
  if (max_error >= 0.0) {
    r = ZL_CCtx_setParameter(cctx, ZL_CParam_contentChecksum,
                             ZL_TernaryParam_disable);
    if (ZL_isError(r))
      goto done;
  }

  in = ZL_TypedRef_createNumeric(src, eltWidth, numElts);
  if (in == NULL) {
    r = ZL_returnError(ZL_ErrorCode_allocation);
    goto done;
  }

  r = ZL_CCtx_compressTypedRef(cctx, dst, dstCapacity, in);
  if (!ZL_isError(r) && outSize != NULL)
    *outSize = ZL_validResult(r);

done:
  if (in != NULL)
    ZL_TypedRef_free(in);
  if (cctx != NULL)
    ZL_CCtx_free(cctx);
  ZL_Compressor_free(c);
  return r;
}