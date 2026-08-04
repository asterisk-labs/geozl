#define _POSIX_C_SOURCE 200809L // clock_gettime

#include "geozl/geozl.h"

#include "openzl/zl_compress.h"
#include "openzl/zl_compressor.h"
#include "openzl/zl_data.h"
#include "openzl/zl_decompress.h"
#include "openzl/zl_errors_types.h"
#include "openzl/zl_graph_api.h"
#include "openzl/zl_version.h" // ZL_MAX_FORMAT_VERSION
#include "openzl/codecs/zl_conversion.h" // ZL_NODE_CONVERT_*
#include "openzl/codecs/zl_delta.h"      // ZL_NODE_DELTA_INT
#include "openzl/codecs/zl_entropy.h"    // ZL_GRAPH_ENTROPY
#include "openzl/codecs/zl_field_lz.h"   // ZL_GRAPH_FIELD_LZ
#include "openzl/codecs/zl_illegal.h"    // ZL_GRAPH_ILLEGAL
#include "openzl/codecs/zl_store.h"      // ZL_GRAPH_STORE
#include "openzl/codecs/zl_transpose.h"  // ZL_NODE_TRANSPOSE_SPLIT
#include "openzl/codecs/zl_zigzag.h"     // ZL_NODE_ZIGZAG
#include "openzl/codecs/zl_zstd.h"       // ZL_GRAPH_ZSTD

#include "lossy/lossy_node.h"             // geozl_node_lossy
#include "lossy/lossy_recipe.h"           // geozl_lossy_parse and friends
#include "nodata/encode_nodata_binding.h" // GEOZL_NODATA_MODE_*

#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// The 2d entry points. A method name and a raster in, a geozl frame out.
//
// A method names a predictor and a terminal, "planar>zigzag>entropy". The graph
// is built inward from that terminal, so a sample runs
//
//   [nodata] -> [quantizer] -> [predictor -> zigzag] -> terminal
//
// Only the terminal is always there. The predictor and its zigzag come as a
// pair, and the id and delta_1d methods carry neither. The quantizer appears
// when error names one. nodata appears when a mode is set, and it is the one
// stage that does not hand a single stream on: it splits the raster from a
// validity mask and sends the mask down a generic graph of its own.
//
// Two of those positions are not free. nodata is outermost because a quantizer
// has no answer for a sample that was never measured. The quantizer sits ahead
// of the predictor so the residual is taken over the values the frame will
// actually carry. Past the terminal everything belongs to OpenZL.
//
// Decompression needs none of it, since a frame names its own codecs.
//
// The file runs in that order: the palette of method names, the graph built
// from one, the error plumbing OpenZL forces, and then the five entry points,
// compress, frame size, decompress, bench and grid.

// GEOZL_PRED_ID is the no-predictor branch: raw stream straight to a backend,
// no zigzag. delta_1d is OpenZL's ZL_NODE_DELTA_INT and carries no zigzag either.
typedef enum {
  GEOZL_PRED_DELTA_W = 0,
  GEOZL_PRED_DELTA_N,
  GEOZL_PRED_PLANAR,
  GEOZL_PRED_MED,
  GEOZL_PRED_AVERAGE,
  GEOZL_PRED_WP_STATIC,
  GEOZL_PRED_DELTA_1D,
  GEOZL_PRED_ID,
  GEOZL_PRED_COUNT
} geozl_predictor;

// A terminal fixes the layout (interleaved or transposed to byte lanes) and the
// backend. The transposed ones send every lane to one backend; store_lo
// transposes but routes each lane on its own.
typedef enum {
  GEOZL_TERM_ENTROPY = 0, // adaptive FSE/Huffman
  GEOZL_TERM_FIELD_LZ,    // field-wise LZ over the numeric residual
  GEOZL_TERM_ZSTD,        // serialize, then zstd
  GEOZL_TERM_T_ENTROPY,   // transpose to lanes, entropy (shuffle + entropy)
  GEOZL_TERM_T_ZSTD,      // transpose to lanes, zstd (shuffle + zstd)
  GEOZL_TERM_STORE_LO,    // transpose, low lane stored, high lanes entropy
  GEOZL_TERM_COUNT
} geozl_terminal;

static const char *pred_name(geozl_predictor p) {
  switch (p) {
  case GEOZL_PRED_DELTA_W:   return "delta_w";
  case GEOZL_PRED_DELTA_N:   return "delta_n";
  case GEOZL_PRED_PLANAR:    return "planar";
  case GEOZL_PRED_MED:       return "med";
  case GEOZL_PRED_AVERAGE:   return "average";
  case GEOZL_PRED_WP_STATIC: return "wp_static";
  case GEOZL_PRED_DELTA_1D:  return "delta_1d";
  case GEOZL_PRED_ID:        return "id";
  default:                   return "?";
  }
}

static const char *term_name(geozl_terminal t) {
  switch (t) {
  case GEOZL_TERM_ENTROPY:   return "entropy";
  case GEOZL_TERM_FIELD_LZ:  return "field_lz";
  case GEOZL_TERM_ZSTD:      return "zstd";
  case GEOZL_TERM_T_ENTROPY: return "transpose>entropy";
  case GEOZL_TERM_T_ZSTD:    return "transpose>zstd";
  case GEOZL_TERM_STORE_LO:  return "store_lo";
  default:                   return "?";
  }
}

// Recipe string of a candidate, e.g. "planar>zigzag>entropy". This is the only
// place the spelling lives; parse_candidate below reads it back.
static void candidate_name(geozl_predictor p, geozl_terminal t, char *out,
                           size_t cap) {
  if (p == GEOZL_PRED_ID)
    snprintf(out, cap, "id>%s", term_name(t));
  else if (p == GEOZL_PRED_DELTA_1D)
    snprintf(out, cap, "delta_1d>%s", term_name(t));
  else
    snprintf(out, cap, "%s>zigzag>%s", pred_name(p), term_name(t));
}

// Inverse of candidate_name. It generates and compares instead of splitting on
// ">", so the two can never drift apart. 48 snprintf on a call that is about to
// compress a whole tile is not worth optimizing.
static int parse_candidate(const char *name, geozl_predictor *outP,
                           geozl_terminal *outT) {
  if (name == NULL || name[0] == '\0')
    return -1;
  char buf[48];
  for (int p = 0; p < GEOZL_PRED_COUNT; ++p) {
    for (int t = 0; t < GEOZL_TERM_COUNT; ++t) {
      candidate_name((geozl_predictor)p, (geozl_terminal)t, buf, sizeof(buf));
      if (strcmp(name, buf) == 0) {
        *outP = (geozl_predictor)p;
        *outT = (geozl_terminal)t;
        return 0;
      }
    }
  }
  return -1;
}

static ZL_NodeID predictor_node(ZL_Compressor *c, geozl_predictor p,
                                uint32_t width) {
  switch (p) {
  case GEOZL_PRED_DELTA_W:   return geozl_node_delta_w(c, width);
  case GEOZL_PRED_DELTA_N:   return geozl_node_delta_n(c, width);
  case GEOZL_PRED_PLANAR:    return geozl_node_planar(c, width);
  case GEOZL_PRED_MED:       return geozl_node_med(c, width);
  case GEOZL_PRED_AVERAGE:   return geozl_node_average(c, width);
  case GEOZL_PRED_WP_STATIC: return geozl_node_wp_static(c, width);
  default:                   return ZL_NODE_ILLEGAL;
  }
}

// n == 0 is the id branch, a backend with no predictor in front of it.
static ZL_GraphID chain(ZL_Compressor *c, const ZL_NodeID *nodes, size_t n,
                        ZL_GraphID final) {
  if (n == 0)
    return final;
  if (n == 1)
    return ZL_Compressor_registerStaticGraph_fromNode1o(c, nodes[0], final);
  return ZL_Compressor_registerStaticGraph_fromPipelineNodes1o(c, nodes, n,
                                                               final);
}

// store_lo splits the struct into byte lanes at run time (TRANSPOSE_SPLIT has a
// variable output count, so it cannot be wired statically) and routes lane 0,
// the residual noise, to a raw store while the near-constant higher lanes go to
// entropy.
static ZL_Report geozl_storelo_fg(ZL_Graph *g, ZL_Edge *inputs[],
                                  size_t nbInputs) {
  (void)nbInputs;
  ZL_RESULT_OF(ZL_EdgeList) lanes = ZL_Edge_runTransposeSplit(inputs[0], g);
  if (ZL_RES_isError(lanes))
    return ZL_returnError(ZL_RES_code(lanes));
  ZL_EdgeList e = ZL_RES_value(lanes);
  for (size_t i = 0; i < e.nbEdges; ++i) {
    ZL_Report r = ZL_Edge_setDestination(
        e.edges[i], i == 0 ? ZL_GRAPH_STORE : ZL_GRAPH_ENTROPY);
    if (ZL_isError(r))
      return r;
  }
  return ZL_returnSuccess();
}

// One candidate graph, or ZL_GRAPH_ILLEGAL if the pair does not apply.
static ZL_GraphID build_candidate(ZL_Compressor *c, geozl_predictor p,
                                  geozl_terminal t, uint32_t width,
                                  size_t eltWidth) {
  ZL_NodeID head[3];
  size_t n = 0;
  if (p == GEOZL_PRED_ID) {
    // no residual: the raw numeric stream feeds the terminal
  } else if (p == GEOZL_PRED_DELTA_1D) {
    head[n++] = ZL_NODE_DELTA_INT;
  } else {
    ZL_NodeID pred = predictor_node(c, p, width);
    if (!ZL_NodeID_isValid(pred))
      return ZL_GRAPH_ILLEGAL;
    head[n++] = pred;
    head[n++] = ZL_NODE_ZIGZAG;
  }

  switch (t) {
  case GEOZL_TERM_ENTROPY:
    return chain(c, head, n, ZL_GRAPH_ENTROPY);
  case GEOZL_TERM_FIELD_LZ:
    return chain(c, head, n, ZL_GRAPH_FIELD_LZ);
  case GEOZL_TERM_ZSTD:
    head[n++] = ZL_NODE_CONVERT_NUM_TO_SERIAL;
    return chain(c, head, n, ZL_GRAPH_ZSTD);
  case GEOZL_TERM_T_ENTROPY:
  case GEOZL_TERM_T_ZSTD: {
    // transpose to byte lanes, every lane to one backend
    if (eltWidth < 2 || eltWidth > 8)
      return ZL_GRAPH_ILLEGAL;
    ZL_GraphID back = (t == GEOZL_TERM_T_ZSTD) ? ZL_GRAPH_ZSTD : ZL_GRAPH_ENTROPY;
    ZL_GraphID tg = ZL_Compressor_registerTransposeSplitGraph(c, back);
    if (!ZL_GraphID_isValid(tg))
      return ZL_GRAPH_ILLEGAL;
    head[n++] = ZL_NODE_CONVERT_NUM_TO_STRUCT_LE;
    return chain(c, head, n, tg);
  }
  case GEOZL_TERM_STORE_LO: {
    // low lane is residual noise, stored raw; higher lanes are near-constant,
    // so entropy (which collapses a constant lane to almost nothing).
    if (eltWidth < 2 || eltWidth > 8)
      return ZL_GRAPH_ILLEGAL;
    static const ZL_Type in_mask = ZL_Type_struct;
    static const ZL_GraphID used[2] = {ZL_GRAPH_STORE, ZL_GRAPH_ENTROPY};
    ZL_FunctionGraphDesc desc = {0};
    desc.name = "geozl_store_lo";
    desc.graph_f = geozl_storelo_fg;
    desc.inputTypeMasks = &in_mask;
    desc.nbInputs = 1;
    desc.customGraphs = used;
    desc.nbCustomGraphs = 2;
    ZL_GraphID fg = ZL_Compressor_registerFunctionGraph(c, &desc);
    if (!ZL_GraphID_isValid(fg))
      return ZL_GRAPH_ILLEGAL;
    head[n++] = ZL_NODE_CONVERT_NUM_TO_STRUCT_LE;
    return chain(c, head, n, fg);
  }
  default:
    return ZL_GRAPH_ILLEGAL;
  }
}

// build_candidate gives the predictor and terminal, this wraps the quantizer
// and then nodata around it, outermost last.
static ZL_GraphID build_graph(ZL_Compressor *c, geozl_predictor p,
                                 geozl_terminal t, uint32_t width,
                                 size_t eltWidth,
                                 const geozl_lossy_plan *plan, int dtype,
                                 int nodataMode, uint64_t nodataBits) {
  ZL_GraphID sel = build_candidate(c, p, t, width, eltWidth);
  if (!ZL_GraphID_isValid(sel))
    return ZL_GRAPH_ILLEGAL;

  if (plan->family != GEOZL_LOSSY_NONE) {
    ZL_NodeID q = geozl_node_lossy(c, plan, dtype);
    if (!ZL_NodeID_isValid(q))
      return ZL_GRAPH_ILLEGAL;
    sel = ZL_Compressor_registerStaticGraph_fromNode1o(c, q, sel);
    if (!ZL_GraphID_isValid(sel))
      return ZL_GRAPH_ILLEGAL;
  }

  if (nodataMode == GEOZL_NODATA_NONE)
    return sel;

  uint64_t bits = 0;
  int mode = GEOZL_NODATA_MODE_NAN;
  if (nodataMode == GEOZL_NODATA_VALUE) {
    bits = (eltWidth == 8)
               ? nodataBits
               : (nodataBits & (((uint64_t)1 << (8 * eltWidth)) - 1));
    mode = GEOZL_NODATA_MODE_VALUE;
  }

  ZL_NodeID nd = geozl_node_nodata(c, width, mode, bits);
  if (!ZL_NodeID_isValid(nd))
    return ZL_GRAPH_ILLEGAL;
  // Outcome 0 is the raster and carries on down the recipe, outcome 1 is the
  // mask.
  return ZL_Compressor_registerStaticGraph_fromNode(
      c, nd, ZL_GRAPHLIST(sel, ZL_GRAPH_COMPRESS_GENERIC));
}

// An OpenZL error string lives inside the object that raised it and only for
// that object's lifetime, so it has to be copied out before the free below. It
// is also rejected by any other object, hence err_owner.
typedef enum { ERR_NONE, ERR_COMPRESSOR, ERR_CCTX, ERR_DCTX } err_owner;

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

// errCtx receives the reason on failure and an empty string on success,
// truncated to errCtxSize. Pass NULL to drop it.
static ZL_Report compress_impl(const char *method, uint32_t width,
                               const char *error, int dtype, int nodataMode,
                               uint64_t nodataBits, const void *src,
                               size_t numElts, size_t eltWidth, void *dst,
                               size_t dstCapacity, size_t *outSize,
                               char *errCtx, size_t errCtxSize, int noChecksum) {
  const int has_err = errCtx != NULL && errCtxSize != 0;
  if (has_err)
    errCtx[0] = '\0';

  // ZL_TypedRef_createNumeric only returns NULL, so an unsupported width would
  // otherwise be reported as an allocation failure.
  if (eltWidth != 1 && eltWidth != 2 && eltWidth != 4 && eltWidth != 8) {
    if (has_err)
      snprintf(errCtx, errCtxSize,
               "eltWidth %zu, an OpenZL numeric stream is 1, 2, 4 or 8",
               eltWidth);
    return ZL_returnError(ZL_ErrorCode_parameter_invalid);
  }

  // Every quantizer indexes tables by this, checked once here.
  if (!GEOZL_DT_OK(dtype)) {
    if (has_err)
      snprintf(errCtx, errCtxSize,
               "dtype %d is outside the geozl_dtype codes, which run 0 to %d",
               dtype, (int)GEOZL_DT_F64);
    return ZL_returnError(ZL_ErrorCode_parameter_invalid);
  }

  // A bad name is a caller mistake, a name that does not apply at this width is
  // a graph problem. Keeping them apart makes the message actionable.
  geozl_predictor pred;
  geozl_terminal term;
  if (parse_candidate(method, &pred, &term) != 0) {
    if (has_err)
      snprintf(errCtx, errCtxSize, "unknown method \"%s\"",
               method != NULL ? method : "(null)");
    return ZL_returnError(ZL_ErrorCode_parameter_invalid);
  }

  // Cut here and not in the node, so compress, bench and profile all report the
  // same numbers. A SQRT recipe with no curve gets one fitted from this tile,
  // since nothing at this entry point carries a product.
  geozl_lossy_recipe recipe;
  geozl_lossy_plan plan;
  {
    char why[192] = {0};
    if (geozl_lossy_parse(error, &recipe, why, sizeof(why)) != 0 ||
        geozl_lossy_fit(&recipe, src, dtype, width, numElts, why,
                        sizeof(why)) != 0 ||
        geozl_lossy_resolve(&recipe, src, dtype, numElts, &plan, why,
                            sizeof(why)) != 0) {
      if (has_err)
        snprintf(errCtx, errCtxSize, "%s", why);
      return ZL_returnError(ZL_ErrorCode_parameter_invalid);
    }
  }

  ZL_Compressor *c = ZL_Compressor_create();
  if (c == NULL)
    return ZL_returnError(ZL_ErrorCode_allocation);

  ZL_Report r;
  err_owner owner = ERR_NONE;
  ZL_TypedRef *in = NULL;
  ZL_CCtx *cctx = NULL;

  ZL_GraphID g = build_graph(c, pred, term, width, eltWidth, &plan, dtype,
                             nodataMode, nodataBits);
  if (!ZL_GraphID_isValid(g)) {
    if (has_err)
      snprintf(errCtx, errCtxSize,
               "method \"%s\" does not apply to %zu-byte elements; the "
               "transpose and store_lo terminals need 2 to 8",
               method, eltWidth);
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
  r = ZL_CCtx_setParameter(cctx, ZL_CParam_formatVersion, ZL_MAX_FORMAT_VERSION);
  if (ZL_isError(r)) {
    owner = ERR_CCTX;
    goto done;
  }
  if (noChecksum || plan.family != GEOZL_LOSSY_NONE) {
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

// The one compression entry. Returns 0 or the ZL_ErrorCode, keeping ZL_Report
// out of the bindings, with the reason in errCtx and the frame size in *outSize.
GEOZL_API int geozl_2d_compress_c(const char *method, uint32_t width,
                                  const char *error, int dtype, int nodataMode,
                                  uint64_t nodataBits, const void *src,
                                  size_t numElts, size_t eltWidth, void *dst,
                                  size_t dstCapacity, size_t *outSize,
                                  char *errCtx, size_t errCtxSize) {
  ZL_Report r = compress_impl(method, width, error, dtype, nodataMode,
                              nodataBits, src, numElts, eltWidth, dst,
                              dstCapacity, outSize, errCtx, errCtxSize, 0);
  return ZL_isError(r) ? (int)ZL_errorCode(r) : 0;
}

// Decompressed byte size of a frame, or 0 if the frame cannot be read. Callers
// size their output buffer with this before geozl_2d_decompress_c.
GEOZL_API size_t geozl_2d_frame_dsize_c(const void *frame, size_t frameSize) {
  ZL_Report r = ZL_getDecompressedSize(frame, frameSize);
  return ZL_isError(r) ? 0 : ZL_validResult(r);
}

// Decompress a geozl frame into dst. Returns 0 on success or the ZL_ErrorCode,
// with the reason in errCtx. The frame is self-describing, so no method,
// predictor or width is needed here.
GEOZL_API int geozl_2d_decompress_c(const void *frame, size_t frameSize,
                                    void *dst, size_t dstCapacity,
                                    size_t *outSize, char *errCtx,
                                    size_t errCtxSize) {
  const int has_err = errCtx != NULL && errCtxSize != 0;
  if (has_err)
    errCtx[0] = '\0';

  ZL_DCtx *dctx = ZL_DCtx_create();
  if (dctx == NULL)
    return (int)ZL_ErrorCode_allocation;

  ZL_Report r = geozl_register_decoders(dctx);
  err_owner owner = ERR_NONE;
  if (ZL_isError(r)) {
    owner = ERR_DCTX;
    goto done;
  }
  // Our frames carry a single numeric output; ZL_DCtx_decompress only returns
  // serial, so the typed variant is required. dst must be 8-byte aligned.
  ZL_OutputInfo info;
  r = ZL_DCtx_decompressTyped(dctx, &info, dst, dstCapacity, frame, frameSize);
  if (ZL_isError(r))
    owner = ERR_DCTX;
  else if (outSize != NULL)
    *outSize = (size_t)info.decompressedByteSize;

done:
  if (owner == ERR_DCTX)
    copy_err(errCtx, errCtxSize, ZL_DCtx_getErrorContextString(dctx, r));
  ZL_DCtx_free(dctx);
  return ZL_isError(r) ? (int)ZL_errorCode(r) : 0;
}

static double now_sec(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

// Time one graph: reps compressions and reps decompressions, all in C so
// the FFI is crossed once, not once per rep. Returns 0, or the ZL_ErrorCode of
// the first failing round trip. compSize gets the frame size, encSec/decSec the
// best (minimum) time of the reps.
GEOZL_API int geozl_2d_bench_c(const char *method, uint32_t width,
                               const char *error, int dtype, int nodataMode,
                               uint64_t nodataBits, const void *src,
                               size_t numElts, size_t eltWidth, size_t reps,
                               size_t *compSize, double *encSec, double *decSec,
                               char *errCtx, size_t errCtxSize) {
  // reps == 0 would leave cbuf unwritten and the timings at DBL_MAX
  if (reps == 0)
    return (int)ZL_ErrorCode_parameter_invalid;

  size_t cap = 1024 + numElts * eltWidth + numElts * eltWidth / 2;
  void *cbuf = malloc(cap);
  if (cbuf == NULL)
    return (int)ZL_ErrorCode_allocation;

  double bestEnc = DBL_MAX;
  size_t sz = 0;
  for (size_t i = 0; i < reps; ++i) {
    double t0 = now_sec();
    ZL_Report r =
        compress_impl(method, width, error, dtype, nodataMode, nodataBits,
                      src, numElts, eltWidth, cbuf, cap, &sz, errCtx,
                      errCtxSize, 1);
    double dt = now_sec() - t0;
    if (ZL_isError(r)) {
      free(cbuf);
      return (int)ZL_errorCode(r);
    }
    if (dt < bestEnc)
      bestEnc = dt;
  }

  size_t dsize = geozl_2d_frame_dsize_c(cbuf, sz);
  void *dbuf = malloc(dsize ? dsize : 1); // malloc is max-aligned, fine for numeric
  if (dbuf == NULL) {
    free(cbuf);
    return (int)ZL_ErrorCode_allocation;
  }

  double bestDec = DBL_MAX;
  for (size_t i = 0; i < reps; ++i) {
    double t0 = now_sec();
    int rc = geozl_2d_decompress_c(cbuf, sz, dbuf, dsize, NULL, errCtx,
                                   errCtxSize);
    double dt = now_sec() - t0;
    if (rc != 0) {
      free(dbuf);
      free(cbuf);
      return rc;
    }
    if (dt < bestDec)
      bestDec = dt;
  }

  if (compSize != NULL)
    *compSize = sz;
  if (encSec != NULL)
    *encSec = bestEnc;
  if (decSec != NULL)
    *decSec = bestDec;
  free(dbuf);
  free(cbuf);
  return 0;
}

// Expands a method string into the predictor list geozl_2d_grid_c enumerates
// for the Python profiler. A predictor name gives that predictor plus the id
// pass; "none" or "id" gives the id pass alone; NULL or "" gives every
// predictor. This does not drive compression, parse_candidate does.
static int resolve_prior(const char *method, geozl_predictor *out,
                         size_t *outN) {
  if (method == NULL || method[0] == '\0') {
    size_t k = 0;
    for (int p = 0; p < GEOZL_PRED_COUNT; ++p)
      out[k++] = (geozl_predictor)p; // includes GEOZL_PRED_ID last
    *outN = k;
    return 0;
  }
  if (strcmp(method, "none") == 0 || strcmp(method, "id") == 0) {
    out[0] = GEOZL_PRED_ID;
    *outN = 1;
    return 0;
  }
  // Matched against pred_name rather than a second table, so the two spellings
  // cannot drift. Same reason parse_candidate generates instead of splitting.
  for (int p = 0; p < GEOZL_PRED_COUNT; ++p) {
    if (p == GEOZL_PRED_ID)
      continue; // handled above, and it takes no second pass
    if (strcmp(method, pred_name((geozl_predictor)p)) == 0) {
      out[0] = (geozl_predictor)p;
      out[1] = GEOZL_PRED_ID;
      *outN = 2;
      return 0;
    }
  }
  return -1;
}

// Recipe names of the grid a method expands to, one per stride-byte slot. Lets
// profile learn the palette without compressing. -1 on an unknown method.
GEOZL_API int geozl_2d_grid_c(const char *method, size_t eltWidth, char *names,
                              size_t stride, size_t maxNames, size_t *outCount) {
  geozl_predictor preds[GEOZL_PRED_COUNT];
  size_t nbPreds = 0;
  if (resolve_prior(method, preds, &nbPreds) != 0)
    return -1;

  size_t k = 0;
  for (size_t i = 0; i < nbPreds; ++i) {
    for (int t = 0; t < GEOZL_TERM_COUNT; ++t) {
      // the transpose-based terminals need at least a 2-byte element
      if ((geozl_terminal)t >= GEOZL_TERM_T_ENTROPY &&
          (eltWidth < 2 || eltWidth > 8))
        continue;
      if (k < maxNames)
        candidate_name(preds[i], (geozl_terminal)t, names + k * stride, stride);
      k++;
    }
  }
  if (outCount != NULL)
    *outCount = k;
  return 0;
}