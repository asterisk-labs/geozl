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

#include <stdio.h>
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
// An OpenZL error string lives inside the object that raised it and only for
// that object's lifetime, so it has to be copied out before the free below. It
// is also rejected by any other object, hence err_owner.
typedef enum { ERR_NONE, ERR_COMPRESSOR, ERR_CCTX } err_owner;

static void copy_err(char *dst, size_t cap, const char *src) {
  if (dst == NULL || cap == 0)
    return;
  if (src == NULL) {
    dst[0] = '\0';
    return;
  }
  size_t n = strlen(src);
  if (n >= cap)
    n = cap - 1;
  memcpy(dst, src, n);
  dst[n] = '\0';
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

// errCtx receives the reason on failure and an empty string on success,
// truncated to errCtxSize. Pass NULL to drop it.
ZL_Report geozl_2d_compress(const char *method, uint32_t width,
                            double max_error, int dtype, int formatVersion,
                            const void *src, size_t numElts, size_t eltWidth,
                            void *dst, size_t dstCapacity, size_t *outSize,
                            char *errCtx, size_t errCtxSize) {
  const int has_err = errCtx != NULL && errCtxSize != 0;
  if (has_err)
    errCtx[0] = '\0';

  geozl_predictor set[GEOZL_PRED_COUNT];
  size_t n = 0;
  if (geozl_2d_method(method, set, &n) != 0) {
    if (has_err)
      snprintf(errCtx, errCtxSize, "unknown method \"%s\"",
               method != NULL ? method : "(null)");
    return ZL_returnError(ZL_ErrorCode_parameter_invalid);
  }

  // ZL_TypedRef_createNumeric only returns NULL, so an unsupported width would
  // otherwise be reported as an allocation failure.
  if (eltWidth != 1 && eltWidth != 2 && eltWidth != 4 && eltWidth != 8) {
    if (has_err)
      snprintf(errCtx, errCtxSize,
               "eltWidth %zu, an OpenZL numeric stream is 1, 2, 4 or 8",
               eltWidth);
    return ZL_returnError(ZL_ErrorCode_parameter_invalid);
  }

  ZL_Compressor *c = ZL_Compressor_create();
  if (c == NULL)
    return ZL_returnError(ZL_ErrorCode_allocation);

  ZL_Report r;
  err_owner owner = ERR_NONE;
  ZL_TypedRef *in = NULL;
  ZL_CCtx *cctx = NULL;

  ZL_GraphID g = geozl_2d_graph(c, set, n, width, max_error, dtype);
  if (!ZL_GraphID_isValid(g)) {
    if (has_err)
      snprintf(errCtx, errCtxSize, "could not build the graph for method \"%s\"",
               method);
    r = ZL_returnError(ZL_ErrorCode_graph_invalid);
    goto done;
  }
  r = ZL_Compressor_selectStartingGraphID(c, g);
  if (ZL_isError(r)) {
    owner = ERR_COMPRESSOR;
    goto done;
  }

  cctx = ZL_CCtx_create();
  if (cctx == NULL) {
    r = ZL_returnError(ZL_ErrorCode_allocation);
    goto done;
  }
  r = ZL_CCtx_refCompressor(cctx, c);
  if (ZL_isError(r)) {
    owner = ERR_CCTX;
    goto done;
  }
  r = ZL_CCtx_setParameter(cctx, ZL_CParam_formatVersion, formatVersion);
  if (ZL_isError(r)) {
    owner = ERR_CCTX;
    goto done;
  }
  if (max_error >= 0.0) {
    r = ZL_CCtx_setParameter(cctx, ZL_CParam_contentChecksum,
                             ZL_TernaryParam_disable);
    if (ZL_isError(r)) {
      owner = ERR_CCTX;
      goto done;
    }
  }

  in = ZL_TypedRef_createNumeric(src, eltWidth, numElts);
  if (in == NULL) {
    if (has_err)
      snprintf(errCtx, errCtxSize, "could not wrap src as a numeric stream");
    r = ZL_returnError(ZL_ErrorCode_allocation);
    goto done;
  }

  r = ZL_CCtx_compressTypedRef(cctx, dst, dstCapacity, in);
  if (ZL_isError(r))
    owner = ERR_CCTX;
  else if (outSize != NULL)
    *outSize = ZL_validResult(r);

done:
  if (owner == ERR_CCTX)
    copy_err(errCtx, errCtxSize, ZL_CCtx_getErrorContextString(cctx, r));
  else if (owner == ERR_COMPRESSOR)
    copy_err(errCtx, errCtxSize, ZL_Compressor_getErrorContextString(c, r));
  if (in != NULL)
    ZL_TypedRef_free(in);
  if (cctx != NULL)
    ZL_CCtx_free(cctx);
  ZL_Compressor_free(c);
  return r;
}


// Binding entry: like geozl_2d_compress but returns 0 on success or the
// ZL_ErrorCode on failure, keeping ZL_Report out of foreign bindings. The
// compressed size lands in *outSize on success.
GEOZL_API int geozl_2d_compress_c(const char *method, uint32_t width,
                                  double max_error, int dtype, int formatVersion,
                                  const void *src, size_t numElts,
                                  size_t eltWidth, void *dst, size_t dstCapacity,
                                  size_t *outSize, char *errCtx,
                                  size_t errCtxSize) {
  ZL_Report r = geozl_2d_compress(method, width, max_error, dtype, formatVersion,
                                  src, numElts, eltWidth, dst, dstCapacity,
                                  outSize, errCtx, errCtxSize);
  return ZL_isError(r) ? (int)ZL_errorCode(r) : 0;
}