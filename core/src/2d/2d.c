#define _POSIX_C_SOURCE 200809L // clock_gettime

#include "geozl/geozl.h"

#include "openzl/zl_compress.h"
#include "openzl/zl_compressor.h"
#include "openzl/zl_data.h"
#include "openzl/zl_decompress.h"
#include "openzl/zl_errors_types.h"
#include "openzl/zl_graph_api.h"
#include "openzl/zl_input.h"
#include "openzl/zl_selector.h"
#include "openzl/zl_version.h" // ZL_MAX_FORMAT_VERSION
#include "openzl/codecs/zl_constant.h"   // ZL_GRAPH_CONSTANT
#include "openzl/codecs/zl_conversion.h" // ZL_NODE_CONVERT_*
#include "openzl/codecs/zl_delta.h"      // ZL_NODE_DELTA_INT
#include "openzl/codecs/zl_entropy.h"    // ZL_GRAPH_ENTROPY
#include "openzl/codecs/zl_field_lz.h"   // ZL_GRAPH_FIELD_LZ
#include "openzl/codecs/zl_illegal.h"    // ZL_GRAPH_ILLEGAL
#include "openzl/codecs/zl_store.h"      // ZL_GRAPH_STORE
#include "openzl/codecs/zl_transpose.h"  // ZL_NODE_TRANSPOSE_SPLIT
#include "openzl/codecs/zl_zigzag.h"     // ZL_NODE_ZIGZAG
#include "openzl/codecs/zl_zstd.h"       // ZL_GRAPH_ZSTD

#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
// backend. The transposed ones send every lane to one backend, the blosc/EOPF
// shape; store_lo transposes but routes each lane on its own.
typedef enum {
  GEOZL_TERM_ENTROPY = 0, // adaptive FSE/Huffman
  GEOZL_TERM_FIELD_LZ,    // field-wise LZ over the numeric residual
  GEOZL_TERM_ZSTD,        // serialize, then zstd
  GEOZL_TERM_T_ENTROPY,   // transpose to lanes, entropy (shuffle + entropy)
  GEOZL_TERM_T_ZSTD,      // transpose to lanes, zstd (shuffle + zstd, EOPF)
  GEOZL_TERM_STORE_LO,    // transpose, low lane stored, high lanes entropy
  GEOZL_TERM_COUNT
} geozl_terminal;

typedef enum {
  GEOZL_OPTIM_STORE = 0,   // smallest frame
  GEOZL_OPTIM_SPEED = 1,   // cheapest decode, size breaks ties
  GEOZL_OPTIM_BALANCED = 2 // frame_norm + lam * decode_norm
} geozl_optim;

#define GEOZL_GRID_MAX (GEOZL_PRED_COUNT * GEOZL_TERM_COUNT)

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

// Recipe string of a candidate, e.g. "planar>zigzag>entropy". graph= matches
// against it; it is generated, never parsed back.
static void candidate_name(geozl_predictor p, geozl_terminal t, char *out,
                           size_t cap) {
  if (p == GEOZL_PRED_ID)
    snprintf(out, cap, "id>%s", term_name(t));
  else if (p == GEOZL_PRED_DELTA_1D)
    snprintf(out, cap, "delta_1d>%s", term_name(t));
  else
    snprintf(out, cap, "%s>zigzag>%s", pred_name(p), term_name(t));
}

// Decode cost in cycles per element. Placeholders until calibrated; only
// balanced and speed read them.
static double pred_decode_cost(geozl_predictor p) {
  switch (p) {
  case GEOZL_PRED_DELTA_W:   return 1.5;
  case GEOZL_PRED_DELTA_N:   return 1.3;
  case GEOZL_PRED_PLANAR:    return 0.9;
  case GEOZL_PRED_MED:       return 4.0;
  case GEOZL_PRED_AVERAGE:   return 3.8;
  case GEOZL_PRED_WP_STATIC: return 2.0;
  case GEOZL_PRED_DELTA_1D:  return 1.0;
  case GEOZL_PRED_ID:        return 0.0;
  default:                   return 0.0;
  }
}

static double term_decode_cost(geozl_terminal t) {
  switch (t) {
  case GEOZL_TERM_ENTROPY:   return 0.5;
  case GEOZL_TERM_FIELD_LZ:  return 0.8;
  case GEOZL_TERM_ZSTD:      return 1.0;
  case GEOZL_TERM_T_ENTROPY: return 0.8; // + transpose
  case GEOZL_TERM_T_ZSTD:    return 1.3; // + transpose
  case GEOZL_TERM_STORE_LO:  return 0.3;
  default:                   return 0.0;
  }
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

// Zero head nodes means the final graph runs on the input directly (the id
// branch reaching a backend with no predictor in front).
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
  ZL_NodeID head[4];
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
    // transpose to byte lanes, every lane to one backend (the blosc/EOPF shape)
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

// Parallel arrays in a fixed order the selector's opaque context mirrors.
typedef struct {
  ZL_GraphID graph[GEOZL_GRID_MAX];
  char name[GEOZL_GRID_MAX][48];
  double cost[GEOZL_GRID_MAX];
  size_t n;
} geozl_grid;

static void build_grid(ZL_Compressor *c, const geozl_predictor *preds,
                       size_t nbPreds, uint32_t width, size_t eltWidth,
                       geozl_grid *grid) {
  grid->n = 0;
  for (size_t i = 0; i < nbPreds; ++i) {
    for (int t = 0; t < GEOZL_TERM_COUNT; ++t) {
      ZL_GraphID g = build_candidate(c, preds[i], (geozl_terminal)t, width,
                                     eltWidth);
      if (!ZL_GraphID_isValid(g))
        continue;
      size_t k = grid->n;
      grid->graph[k] = g;
      candidate_name(preds[i], (geozl_terminal)t, grid->name[k],
                     sizeof(grid->name[k]));
      grid->cost[k] = pred_decode_cost(preds[i]) +
                      term_decode_cost((geozl_terminal)t);
      grid->n++;
    }
  }
}

// OpenZL owns this for the compressor's lifetime and frees it via freeFn, so it
// is heap allocated. Costs and names are aligned to customGraphs order. chosen,
// when set, is a caller buffer the selector writes the winning recipe into.
typedef struct {
  int optim;
  double lam;
  size_t n;
  double cost[GEOZL_GRID_MAX];
  char name[GEOZL_GRID_MAX][48];
  char *chosen;
  size_t chosenCap;
} geozl_sel_ctx;

static void geozl_sel_free(void *unused, void *ptr) {
  (void)unused;
  free(ptr);
}

// OpenZL's brute-force selector, but "best" is set by optim instead of size
// alone. tryGraph gives real bytes; decode is modeled from the cost table.
static ZL_GraphID geozl_select(const ZL_Selector *sel, const ZL_Input *in,
                               const ZL_GraphID *cand, size_t nbCand) {
  const geozl_sel_ctx *ctx = ZL_Selector_getOpaquePtr(sel);
  size_t nElts = ZL_Input_numElts(in);

  size_t bytes[GEOZL_GRID_MAX];
  int ok[GEOZL_GRID_MAX];
  size_t minBytes = SIZE_MAX;
  double minDec = DBL_MAX;
  for (size_t i = 0; i < nbCand; ++i) {
    ZL_GraphReport gr = ZL_Selector_tryGraph(sel, in, cand[i]);
    ok[i] = !ZL_isError(gr.finalCompressedSize);
    if (!ok[i])
      continue;
    bytes[i] = ZL_validResult(gr.finalCompressedSize);
    double dec = ctx->cost[i] * (double)nElts;
    if (bytes[i] < minBytes)
      minBytes = bytes[i];
    if (dec < minDec)
      minDec = dec;
  }
  if (minBytes == 0)
    minBytes = 1;

  int have = 0;
  size_t best = 0;
  double bestScore = 0.0;
  for (size_t i = 0; i < nbCand; ++i) {
    if (!ok[i])
      continue;
    double dec = ctx->cost[i] * (double)nElts;
    double score;
    if (ctx->optim == GEOZL_OPTIM_STORE) {
      score = (double)bytes[i];
    } else if (ctx->optim == GEOZL_OPTIM_SPEED) {
      score = dec;
    } else {
      // minDec 0 (uncalibrated costs) drops the decode term, so balanced
      // degrades to store rather than dividing by zero.
      double decNorm = minDec > 0.0 ? dec / minDec : 0.0;
      score = (double)bytes[i] / (double)minBytes + ctx->lam * decNorm;
    }
    if (!have || score < bestScore ||
        (score == bestScore && bytes[i] < bytes[best])) {
      bestScore = score;
      best = i;
      have = 1;
    }
  }
  size_t pick = have ? best : 0;
  if (ctx->chosen != NULL && ctx->chosenCap != 0)
    snprintf(ctx->chosen, ctx->chosenCap, "%s", ctx->name[pick]);
  return cand[pick];
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

// A named predictor seeds {predictor, id}. "none" is the id branch alone.
// NULL or "" is unbiased: every predictor plus id.
static int resolve_prior(const char *method, geozl_predictor *out,
                         size_t *outN) {
  static const struct {
    const char *name;
    geozl_predictor id;
  } named[] = {
      {"delta_w", GEOZL_PRED_DELTA_W}, {"delta_n", GEOZL_PRED_DELTA_N},
      {"planar", GEOZL_PRED_PLANAR},   {"med", GEOZL_PRED_MED},
      {"average", GEOZL_PRED_AVERAGE}, {"wp_static", GEOZL_PRED_WP_STATIC},
      {"delta_1d", GEOZL_PRED_DELTA_1D},
  };

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
  for (size_t i = 0; i < sizeof(named) / sizeof(named[0]); ++i) {
    if (strcmp(method, named[i].name) == 0) {
      out[0] = named[i].id;
      out[1] = GEOZL_PRED_ID;
      *outN = 2;
      return 0;
    }
  }
  return -1;
}

// With graph set, that candidate runs directly and no search happens.
// Otherwise the grid is wrapped in the selector. quant_linear is prepended for
// lossy.
static ZL_GraphID geozl_2d_graph(ZL_Compressor *c,
                                 const geozl_predictor *preds, size_t nbPreds,
                                 const char *graph, int optim, double lam,
                                 uint32_t width, size_t eltWidth,
                                 double max_error, int dtype, char *chosenGraph,
                                 size_t chosenGraphSize) {
  geozl_grid grid;
  build_grid(c, preds, nbPreds, width, eltWidth, &grid);
  if (grid.n == 0)
    return ZL_GRAPH_ILLEGAL;

  ZL_GraphID sel;
  if (graph != NULL && graph[0] != '\0') {
    size_t hit = grid.n;
    for (size_t i = 0; i < grid.n; ++i) {
      if (strcmp(graph, grid.name[i]) == 0) {
        hit = i;
        break;
      }
    }
    if (hit == grid.n)
      return ZL_GRAPH_ILLEGAL;
    if (chosenGraph != NULL && chosenGraphSize != 0)
      snprintf(chosenGraph, chosenGraphSize, "%s", grid.name[hit]);
    sel = grid.graph[hit];
  } else if (grid.n == 1) {
    if (chosenGraph != NULL && chosenGraphSize != 0)
      snprintf(chosenGraph, chosenGraphSize, "%s", grid.name[0]);
    sel = grid.graph[0];
  } else {
    geozl_sel_ctx *ctx = (geozl_sel_ctx *)malloc(sizeof(*ctx));
    if (ctx == NULL)
      return ZL_GRAPH_ILLEGAL;
    ctx->optim = optim;
    ctx->lam = lam;
    ctx->n = grid.n;
    ctx->chosen = chosenGraph;
    ctx->chosenCap = chosenGraphSize;
    memcpy(ctx->cost, grid.cost, grid.n * sizeof(ctx->cost[0]));
    for (size_t i = 0; i < grid.n; ++i)
      snprintf(ctx->name[i], sizeof(ctx->name[i]), "%s", grid.name[i]);

    ZL_SelectorDesc desc = {0};
    desc.selector_f = geozl_select;
    desc.inStreamType = ZL_Type_numeric;
    desc.customGraphs = grid.graph;
    desc.nbCustomGraphs = grid.n;
    desc.name = "geozl_2d_selector";
    desc.opaque.ptr = ctx;
    desc.opaque.freeFn = geozl_sel_free;

    ZL_RESULT_OF(ZL_GraphID) res =
        ZL_Compressor_registerSelectorGraph2(c, &desc);
    if (ZL_RES_isError(res)) {
      free(ctx);
      return ZL_GRAPH_ILLEGAL;
    }
    sel = ZL_RES_value(res);
  }

  if (max_error < 0.0)
    return sel;
  ZL_NodeID q = geozl_node_quant_linear(c, max_error, dtype);
  if (!ZL_NodeID_isValid(q))
    return ZL_GRAPH_ILLEGAL;
  return ZL_Compressor_registerStaticGraph_fromNode1o(c, q, sel);
}

// errCtx receives the reason on failure and an empty string on success,
// truncated to errCtxSize. Pass NULL to drop it.
static ZL_Report compress_impl(const char *method, const char *graph,
                               int optim, double lam, uint32_t width,
                               double max_error, int dtype, const void *src,
                               size_t numElts, size_t eltWidth, void *dst,
                               size_t dstCapacity, size_t *outSize,
                               char *chosenGraph, size_t chosenGraphSize,
                               char *errCtx, size_t errCtxSize, int noChecksum) {
  const int has_err = errCtx != NULL && errCtxSize != 0;
  if (has_err)
    errCtx[0] = '\0';
  if (chosenGraph != NULL && chosenGraphSize != 0)
    chosenGraph[0] = '\0';

  geozl_predictor preds[GEOZL_PRED_COUNT];
  size_t nbPreds = 0;
  if (resolve_prior(method, preds, &nbPreds) != 0) {
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

  ZL_GraphID g = geozl_2d_graph(c, preds, nbPreds, graph, optim, lam, width,
                                eltWidth, max_error, dtype, chosenGraph,
                                chosenGraphSize);
  if (!ZL_GraphID_isValid(g)) {
    if (has_err) {
      if (graph != NULL && graph[0] != '\0')
        snprintf(errCtx, errCtxSize, "unknown or unbuildable graph \"%s\"",
                 graph);
      else
        snprintf(errCtx, errCtxSize, "could not build the grid for method \"%s\"",
                 method != NULL ? method : "(default)");
    }
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
  if (noChecksum || max_error >= 0.0) {
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

// errCtx receives the reason on failure, chosenGraph the winning recipe.
ZL_Report geozl_2d_compress(const char *method, const char *graph, int optim,
                            double lam, uint32_t width, double max_error,
                            int dtype, const void *src, size_t numElts,
                            size_t eltWidth, void *dst, size_t dstCapacity,
                            size_t *outSize, char *chosenGraph,
                            size_t chosenGraphSize, char *errCtx,
                            size_t errCtxSize) {
  return compress_impl(method, graph, optim, lam, width, max_error, dtype, src,
                       numElts, eltWidth, dst, dstCapacity, outSize, chosenGraph,
                       chosenGraphSize, errCtx, errCtxSize, 0);
}

// Binding entry: like geozl_2d_compress but returns 0 on success or the
// ZL_ErrorCode on failure, keeping ZL_Report out of foreign bindings. The
// compressed size lands in *outSize on success.
GEOZL_API int geozl_2d_compress_c(const char *method, const char *graph,
                                  int optim, double lam, uint32_t width,
                                  double max_error, int dtype, const void *src,
                                  size_t numElts, size_t eltWidth, void *dst,
                                  size_t dstCapacity, size_t *outSize,
                                  char *chosenGraph, size_t chosenGraphSize,
                                  char *errCtx, size_t errCtxSize) {
  ZL_Report r = geozl_2d_compress(method, graph, optim, lam, width, max_error,
                                  dtype, src, numElts, eltWidth, dst, dstCapacity,
                                  outSize, chosenGraph, chosenGraphSize, errCtx,
                                  errCtxSize);
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

// Time one pinned graph: reps compressions and reps decompressions, all in C so
// the FFI is crossed once, not once per rep. Returns 0, or the ZL_ErrorCode of
// the first failing round trip. compSize gets the frame size, encSec/decSec the
// best (minimum) time of the reps.
GEOZL_API int geozl_2d_bench_c(const char *method, const char *graph,
                               uint32_t width, double max_error, int dtype,
                               const void *src, size_t numElts, size_t eltWidth,
                               size_t reps, size_t *compSize, double *encSec,
                               double *decSec, char *errCtx, size_t errCtxSize) {
  size_t cap = 1024 + numElts * eltWidth + numElts * eltWidth / 2;
  void *cbuf = malloc(cap);
  if (cbuf == NULL)
    return (int)ZL_ErrorCode_allocation;

  double bestEnc = DBL_MAX;
  size_t sz = 0;
  for (size_t i = 0; i < reps; ++i) {
    double t0 = now_sec();
    ZL_Report r = compress_impl(method, graph, GEOZL_OPTIM_STORE, 0.0, width,
                                max_error, dtype, src, numElts, eltWidth, cbuf,
                                cap, &sz, NULL, 0, errCtx, errCtxSize, 1);
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