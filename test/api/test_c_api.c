// The parts of geozl/geozl.h a C caller reaches without the 2d entry points,
// which need OpenZL linked in and so cannot run from the Python tests, and the
// 2d argument checks no binding can pass.

#include "geozl/dtype.h"
#include "geozl/geozl.h"

#include "openzl/zl_compress.h"
#include "openzl/zl_compressor.h"
#include "openzl/zl_errors.h"
#include "openzl/zl_input.h"
#include "openzl/codecs/zl_entropy.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);                 \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

#define W 256
#define H 256
#define N (W * H)

static uint16_t src[N];
static uint16_t back[N];

// Counts with variance close to their mean over a wide range of means, the
// texture the SQRT fit needs.
static void fill(void) {
  uint32_t s = 0x2468aceu;
  for (size_t y = 0; y < H; ++y)
    for (size_t x = 0; x < W; ++x) {
      const double mean = 50.0 + 8.0 * (double)x + 2.0 * (double)y;
      double g = -6.0;
      for (int k = 0; k < 12; ++k) {
        s = s * 1664525u + 1013904223u;
        g += (double)(s >> 8) / 16777216.0;
      }
      src[y * W + x] = (uint16_t)lround(mean + g * sqrt(mean));
    }
}

// The frame a hand built quantizer -> entropy graph writes. Frees c.
static size_t compress_graph(ZL_Compressor *c, ZL_NodeID q, void *dst,
                             size_t cap) {
  ZL_GraphID g =
      ZL_Compressor_registerStaticGraph_fromNode1o(c, q, ZL_GRAPH_ENTROPY);
  CHECK(!ZL_isError(ZL_Compressor_selectStartingGraphID(c, g)));
  ZL_CCtx *cctx = ZL_CCtx_create();
  CHECK(!ZL_isError(ZL_CCtx_refCompressor(cctx, c)));
  CHECK(!ZL_isError(ZL_CCtx_setParameter(cctx, ZL_CParam_formatVersion, 27)));
  CHECK(!ZL_isError(ZL_CCtx_setParameter(cctx, ZL_CParam_contentChecksum,
                                         ZL_TernaryParam_disable)));
  ZL_TypedRef *in = ZL_TypedRef_createNumeric(src, sizeof(uint16_t), N);
  ZL_Report r = ZL_CCtx_compressTypedRef(cctx, dst, cap, in);
  ZL_TypedRef_free(in);
  ZL_CCtx_free(cctx);
  ZL_Compressor_free(c);
  CHECK(!ZL_isError(r));
  return ZL_isError(r) ? 0 : ZL_validResult(r);
}

static size_t compress_node(const char *recipe, void *dst, size_t cap) {
  char err[256] = {0};
  ZL_Compressor *c = ZL_Compressor_create();
  ZL_NodeID q = geozl_node_quant(c, recipe, W, GEOZL_DT_U16, src, N, err,
                                 sizeof err);
  CHECK(ZL_NodeID_isValid(q));
  if (!ZL_NodeID_isValid(q)) {
    printf("  %s: %s\n", recipe, err);
    ZL_Compressor_free(c);
    return 0;
  }
  return compress_graph(c, q, dst, cap);
}

// Within the bound the recipe declares, so the node used the resolved grid.
static double worst(const char *recipe) {
  double w = 0.0;
  for (size_t i = 0; i < N; ++i) {
    const double x = src[i], y = back[i], d = fabs(x - y);
    double r = d;
    if (strncmp(recipe, "LOG", 3) == 0)
      r = d / (0.01 * x);
    w = r > w ? r : w;
  }
  return w;
}

static void node_quant_matches_the_2d_path(const char *recipe, double limit) {
  const size_t cap = geozl_2d_compress_bound_c(sizeof src, 0);
  unsigned char *a = malloc(cap), *b = malloc(cap);
  const size_t na = compress_node(recipe, a, cap);
  char err[256] = {0};
  size_t nb = 0;
  CHECK(geozl_2d_compress_c("id>entropy", W, 1, recipe, GEOZL_DT_U16,
                            GEOZL_NODATA_NONE, 0, src, N, sizeof(uint16_t), b,
                            cap, &nb, err, sizeof err) == 0);
  if (err[0] != '\0')
    printf("  %s: %s\n", recipe, err);
  size_t dsize = 1;
  CHECK(na != 0 && geozl_2d_frame_dsize_c(a, na, &dsize) == 0 &&
        dsize == sizeof src);
  size_t got = 0;
  CHECK(geozl_2d_decompress_c(a, na, back, sizeof back, &got, 1, err,
                              sizeof err) == 0 &&
        got == sizeof src);
  CHECK(worst(recipe) <= limit);
  // The 2d path writes one frame per tile, and both decode alike.
  static uint16_t other[N];
  CHECK(geozl_2d_decompress_c(b, nb, other, sizeof other, &got, 1, err,
                              sizeof err) == 0);
  CHECK(memcmp(other, back, sizeof back) == 0);
  free(a);
  free(b);
}

static void node_quant_refuses(const char *recipe, int dtype,
                               const char *needle) {
  char err[256] = {0};
  ZL_Compressor *c = ZL_Compressor_create();
  ZL_NodeID q = geozl_node_quant(c, recipe, W, dtype, src, N, err, sizeof err);
  CHECK(!ZL_NodeID_isValid(q));
  CHECK(strstr(err, needle) != NULL);
  if (strstr(err, needle) == NULL)
    printf("  %s: got \"%s\"\n", recipe ? recipe : "(null)", err);
  ZL_Compressor_free(c);
}

// Each on a fresh compressor, so one refusal cannot stand in for another.
static int builds_linear(unsigned char flags, double step, int dtype) {
  ZL_Compressor *c = ZL_Compressor_create();
  const quant_linear_params p = {.flags = flags, .step = step};
  const int ok = ZL_NodeID_isValid(geozl_node_quant_linear(c, &p, dtype));
  ZL_Compressor_free(c);
  return ok;
}

static int builds_log(unsigned char flags, double step, int dtype) {
  ZL_Compressor *c = ZL_Compressor_create();
  const quant_log_params p = {.flags = flags, .step = step};
  const int ok = ZL_NodeID_isValid(geozl_node_quant_log(c, &p, dtype));
  ZL_Compressor_free(c);
  return ok;
}

static int builds_sqrt(unsigned char flags, double step, double offset,
                       int dtype) {
  ZL_Compressor *c = ZL_Compressor_create();
  const quant_sqrt_params p = {.flags = flags, .step = step, .offset = offset};
  const int ok = ZL_NodeID_isValid(geozl_node_quant_sqrt(c, &p, dtype));
  ZL_Compressor_free(c);
  return ok;
}

// What the kernels would reject at compress time, refused by the builder.
static void quant_builders_refuse_bad_input(void) {
  const quant_linear_params lin = {.flags = 0, .step = 4.0};
  const quant_log_params lg = {.flags = QUANT_LOG_FLAG_STORE_VALUES,
                               .step = 0.02};
  const quant_sqrt_params sq = {.flags = 0, .step = 0.5, .offset = 0.0};
  ZL_Compressor *c = ZL_Compressor_create();
  CHECK(!ZL_NodeID_isValid(geozl_node_quant_linear(NULL, &lin, GEOZL_DT_U16)));
  CHECK(!ZL_NodeID_isValid(geozl_node_quant_linear(c, NULL, GEOZL_DT_U16)));
  CHECK(!ZL_NodeID_isValid(geozl_node_quant_log(NULL, &lg, GEOZL_DT_U16)));
  CHECK(!ZL_NodeID_isValid(geozl_node_quant_log(c, NULL, GEOZL_DT_U16)));
  CHECK(!ZL_NodeID_isValid(geozl_node_quant_sqrt(NULL, &sq, GEOZL_DT_U16)));
  CHECK(!ZL_NodeID_isValid(geozl_node_quant_sqrt(c, NULL, GEOZL_DT_U16)));
  ZL_Compressor_free(c);

  CHECK(builds_linear(0, 4.0, GEOZL_DT_U16));
  CHECK(builds_linear(0, 0.25, GEOZL_DT_F32));
  CHECK(!builds_linear(0, 0.0, GEOZL_DT_U16));
  CHECK(!builds_linear(0, NAN, GEOZL_DT_F32));
  CHECK(!builds_linear(0, 2.5, GEOZL_DT_U16)); // integer grids take whole steps
  CHECK(!builds_linear(0x80, 4.0, GEOZL_DT_U16));
  CHECK(!builds_linear(0, 4.0, 99));
  CHECK(!builds_linear(0, 4.0, -1));

  CHECK(builds_log(QUANT_LOG_FLAG_STORE_VALUES, 0.02, GEOZL_DT_U16));
  CHECK(builds_log(0, 0.02, GEOZL_DT_F32));
  CHECK(!builds_log(QUANT_LOG_FLAG_STORE_VALUES, 0.0, GEOZL_DT_U16));
  CHECK(!builds_log(QUANT_LOG_FLAG_STORE_VALUES, INFINITY, GEOZL_DT_U16));
  CHECK(!builds_log(0, 0.02, GEOZL_DT_U16)); // integers store values
  CHECK(!builds_log(QUANT_LOG_FLAG_STORE_VALUES, 0.02, GEOZL_DT_F32));
  CHECK(!builds_log(QUANT_LOG_FLAG_STORE_VALUES, 0.02, 99));

  CHECK(builds_sqrt(0, 0.5, 0.0, GEOZL_DT_U16));
  CHECK(builds_sqrt(QUANT_SQRT_FLAG_DECODE_F32, 0.5, 4.0, GEOZL_DT_F32));
  CHECK(!builds_sqrt(0, 0.0, 0.0, GEOZL_DT_U16));
  CHECK(!builds_sqrt(0, 1e-200, 0.0, GEOZL_DT_U16)); // squares to a subnormal
  CHECK(!builds_sqrt(0, 0.5, -1.0, GEOZL_DT_U16));
  CHECK(!builds_sqrt(QUANT_SQRT_FLAG_DECODE_F32, 0.5, 0.0, GEOZL_DT_U16));
  CHECK(!builds_sqrt(0, 0.5, 0.0, 99));
}

// OpenZL would dereference a NULL compressor, so every builder refuses one.
static void builders_refuse_a_null_compressor(void) {
  CHECK(!ZL_NodeID_isValid(geozl_node_delta_w(NULL, W)));
  CHECK(!ZL_NodeID_isValid(geozl_node_delta_n(NULL, W, 1)));
  CHECK(!ZL_NodeID_isValid(geozl_node_planar(NULL, W, 1)));
  CHECK(!ZL_NodeID_isValid(geozl_node_planar_zigzag(NULL, W, 1)));
  CHECK(!ZL_NodeID_isValid(geozl_node_planar_zigzag_pfor(NULL, W, 1)));
  CHECK(!ZL_NodeID_isValid(geozl_node_planar_zigzag_lanes(NULL, W, 1)));
  CHECK(!ZL_NodeID_isValid(geozl_node_lanes(NULL)));
  CHECK(!ZL_NodeID_isValid(geozl_node_med(NULL, W, 1)));
  CHECK(!ZL_NodeID_isValid(geozl_node_med_zigzag(NULL, W, 1)));
  CHECK(!ZL_NodeID_isValid(geozl_node_average(NULL, W, 1)));
  CHECK(!ZL_NodeID_isValid(geozl_node_wp_static(NULL, W, 1)));
  CHECK(!ZL_NodeID_isValid(geozl_node_deinterleave(NULL)));
  CHECK(!ZL_NodeID_isValid(geozl_node_pfor(NULL)));
  CHECK(!ZL_NodeID_isValid(
      geozl_node_nodata(NULL, W, GEOZL_NODATA_NAN, 0, GEOZL_DT_F32, 0.0)));
}

// Compress src through q and decode it into back. Frees c.
static int decodes(ZL_Compressor *c, ZL_NodeID q) {
  if (!ZL_NodeID_isValid(q)) {
    ZL_Compressor_free(c);
    return 0;
  }
  const size_t cap = geozl_2d_compress_bound_c(sizeof src, 0);
  unsigned char *frame = malloc(cap);
  const size_t n = compress_graph(c, q, frame, cap);
  char err[256] = {0};
  size_t got = 0;
  const int ok = n != 0 &&
                 geozl_2d_decompress_c(frame, n, back, sizeof back, &got, 1,
                                       err, sizeof err) == 0 &&
                 got == sizeof src;
  if (!ok)
    printf("  decode: %s\n", err);
  free(frame);
  return ok;
}

// back moved off src, so the node quantized, and by no more than
// a + b*x + c*sqrt(x + o), one level of the grid it was given.
static void moved_within(double a, double b, double c, double o) {
  int moved = 0;
  double over = 0.0;
  for (size_t i = 0; i < N; ++i) {
    const double x = src[i], d = fabs(x - back[i]);
    moved |= d != 0.0;
    over = fmax(over, d - (a + b * x + c * sqrt(x + o)));
  }
  CHECK(moved);
  CHECK(over <= 0.0);
}

// Parameters that pass build a node whose frame decodes on its grid.
static void quant_builders_take_valid_parameters(void) {
  const quant_linear_params lin = {.flags = 0, .step = 4.0};
  ZL_Compressor *c = ZL_Compressor_create();
  CHECK(decodes(c, geozl_node_quant_linear(c, &lin, GEOZL_DT_U16)));
  moved_within(0.5 * lin.step, 0.0, 0.0, 0.0);

  const quant_log_params lg = {.flags = QUANT_LOG_FLAG_STORE_VALUES,
                               .step = 0.02};
  c = ZL_Compressor_create();
  CHECK(decodes(c, geozl_node_quant_log(c, &lg, GEOZL_DT_U16)));
  moved_within(1.0, exp2(lg.step) - 1.0, 0.0, 0.0);

  const quant_sqrt_params sq = {.flags = 0, .step = 0.5, .offset = 4.0};
  c = ZL_Compressor_create();
  CHECK(decodes(c, geozl_node_quant_sqrt(c, &sq, GEOZL_DT_U16)));
  moved_within(sq.step * sq.step + 0.5, 0.0, sq.step, sq.offset);
}

static void frame_dsize_tells_empty_from_unreadable(void) {
  unsigned char frame[256];
  char err[256] = {0};
  size_t n = 0;
  CHECK(geozl_2d_compress_c("id>entropy", 1, 1, NULL, GEOZL_DT_U16,
                            GEOZL_NODATA_NONE, 0, src, 0, sizeof(uint16_t),
                            frame, sizeof frame, &n, err, sizeof err) == 0);
  size_t dsize = 7;
  CHECK(geozl_2d_frame_dsize_c(frame, n, &dsize) == 0 && dsize == 0);
  const unsigned char junk[16] = {1, 2, 3, 4};
  dsize = 7;
  CHECK(geozl_2d_frame_dsize_c(junk, sizeof junk, &dsize) != 0 && dsize == 0);
  CHECK(geozl_2d_frame_dsize_c(junk, 0, NULL) != 0);
}

static void compress_bound_covers_noise(void) {
  CHECK(geozl_2d_compress_bound_c(SIZE_MAX, 0) == 0);
  CHECK(geozl_2d_compress_bound_c(0, SIZE_MAX) == 0);
  CHECK(geozl_2d_compress_bound_c(0, 0) > 0);
  CHECK(geozl_2d_compress_bound_c(100, 10) ==
        geozl_2d_compress_bound_c(100, 0) + 14);
  uint32_t s = 7;
  uint64_t *noise = malloc(N * sizeof(uint64_t));
  for (size_t i = 0; i < N; ++i) {
    s = s * 1664525u + 1013904223u;
    noise[i] = ((uint64_t)s << 32) ^ (s * 2654435761u);
  }
  const size_t cap = geozl_2d_compress_bound_c(N * sizeof(uint64_t), 0);
  unsigned char *dst = malloc(cap);
  char err[256] = {0};
  size_t n = 0;
  CHECK(geozl_2d_compress_c("planar>zigzag>zstd", W, 1, NULL, GEOZL_DT_U64,
                            GEOZL_NODATA_NONE, 0, noise, N, sizeof(uint64_t),
                            dst, cap, &n, err, sizeof err) == 0);
  CHECK(n <= cap);
  free(dst);
  free(noise);
}

// A refusal before OpenZL runs, with its reason in err.
#define REFUSED(rc, err, needle)                                               \
  do {                                                                         \
    CHECK((rc) == ZL_ErrorCode_parameter_invalid);                             \
    CHECK(strstr((err), (needle)) != NULL);                                    \
    if (strstr((err), (needle)) == NULL)                                       \
      printf("  wanted \"%s\", got \"%s\"\n", (needle), (err));                \
  } while (0)

static uint64_t any[N]; // aligned for every width, read as any dtype

static const int dtype_of_width[9] = {-1, GEOZL_DT_U8, GEOZL_DT_U16, -1,
                                      GEOZL_DT_U32, -1, -1, -1, GEOZL_DT_U64};

// NaN mode reads samples as floats, so NaN-shaped integers became holes outside
// a LINEAR bound. Value mode reads the sentinel at the dtype's width.
static void nodata_modes_need_their_dtype(void) {
  char err[256];
  for (int d = GEOZL_DT_U8; d <= GEOZL_DT_LAST_INT; ++d) {
    const char *recipes[2] = {NULL, "LINEAR:MAX_ERROR=2"};
    for (int k = 0; k < 2; ++k) {
      geozl_2d_graph *g = (geozl_2d_graph *)any;
      const int rc = geozl_2d_graph_open_c(
          &g, "planar>zigzag>pfor", W, 1, recipes[k], d, GEOZL_NODATA_NAN, 0,
          any, N, geozl_dtype_width(d), err, sizeof err);
      REFUSED(rc, err, "NaN nodata needs a float dtype");
      CHECK(g == NULL);
    }
  }
  for (int d = GEOZL_DT_F16; d <= GEOZL_DT_F64; ++d) {
    geozl_2d_graph *g = NULL;
    CHECK(geozl_2d_graph_open_c(&g, "id>zstd", W, 1, NULL, d,
                                GEOZL_NODATA_NAN, 0, any, N,
                                geozl_dtype_width(d), err, sizeof err) == 0);
    geozl_2d_graph_close_c(g);
  }
  geozl_2d_graph *g = NULL;
  int rc = geozl_2d_graph_open_c(&g, "id>zstd", W, 1, NULL, GEOZL_DT_U8,
                                 GEOZL_NODATA_VALUE, 5, any, N, 2, err,
                                 sizeof err);
  REFUSED(rc, err, "a 1-byte type, but eltWidth is 2");
  CHECK(g == NULL);
  rc = geozl_2d_graph_open_c(&g, "id>zstd", W, 1, NULL, GEOZL_DT_U16,
                             GEOZL_NODATA_VALUE, 5, any, N, 2, err, sizeof err);
  CHECK(rc == 0);
  geozl_2d_graph_close_c(g);
}

static void null_arguments_are_refused(void) {
  static unsigned char frame[4096];
  char err[256];
  size_t n = 99;
  CHECK(geozl_2d_graph_open_c(NULL, "id>zstd", W, 1, NULL, GEOZL_DT_U16,
                              GEOZL_NODATA_NONE, 0, src, N, 2, err,
                              sizeof err) == ZL_ErrorCode_parameter_invalid);
  CHECK(strstr(err, "out is NULL") != NULL);
  int rc = geozl_2d_compress_graph_c(NULL, src, N, frame, sizeof frame, &n, err,
                                     sizeof err);
  REFUSED(rc, err, "graph is NULL");
  CHECK(n == 0);
  n = 99;
  rc = geozl_2d_compress_coeffs_c(NULL, src, N, NULL, 0, frame, sizeof frame,
                                  &n, err, sizeof err);
  REFUSED(rc, err, "graph is NULL");
  CHECK(n == 0);

  // OpenZL would read the samples, or count past size_t, before refusing.
  geozl_2d_graph *g = (geozl_2d_graph *)any;
  rc = geozl_2d_graph_open_c(&g, "id>zstd", W, 1, NULL, GEOZL_DT_U16,
                             GEOZL_NODATA_NONE, 0, NULL, 5, 2, err, sizeof err);
  REFUSED(rc, err, "src is NULL but numElts is 5");
  CHECK(g == NULL);
  rc = geozl_2d_graph_open_c(&g, "id>zstd", W, 1, NULL, GEOZL_DT_U16,
                             GEOZL_NODATA_NONE, 0, src, SIZE_MAX / 2 + 1, 2,
                             err, sizeof err);
  REFUSED(rc, err, "overflow size_t");
  n = 99;
  rc = geozl_2d_compress_c("id>zstd", W, 1, NULL, GEOZL_DT_U16,
                           GEOZL_NODATA_NONE, 0, NULL, 5, 2, frame,
                           sizeof frame, &n, err, sizeof err);
  REFUSED(rc, err, "src is NULL");
  CHECK(n == 0);
  size_t comp = 0;
  double enc = 0, dec = 0;
  rc = geozl_2d_bench_c("id>zstd", W, 1, NULL, GEOZL_DT_U16, GEOZL_NODATA_NONE,
                        0, NULL, 5, 2, 1, 1, 1, &comp, &enc, &dec, err,
                        sizeof err);
  REFUSED(rc, err, "src is NULL");

  CHECK(geozl_2d_graph_open_c(&g, "id>zstd", W, 1, NULL, GEOZL_DT_U16,
                              GEOZL_NODATA_NONE, 0, src, N, 2, err,
                              sizeof err) == 0);
  rc = geozl_2d_compress_graph_c(g, NULL, 5, frame, sizeof frame, &n, err,
                                 sizeof err);
  REFUSED(rc, err, "src is NULL");
  rc = geozl_2d_compress_graph_c(g, src, SIZE_MAX / 2 + 1, frame, sizeof frame,
                                 &n, err, sizeof err);
  REFUSED(rc, err, "overflow size_t");
  rc = geozl_2d_compress_graph_c(g, src, N, NULL, 64, &n, err, sizeof err);
  REFUSED(rc, err, "dst is NULL");
  // OpenZL takes a misaligned stream for an allocation failure.
  rc = geozl_2d_compress_graph_c(g, (const char *)src + 1, 16, frame,
                                 sizeof frame, &n, err, sizeof err);
  REFUSED(rc, err, "not aligned");
  geozl_2d_graph_close_c(g);

  n = 99;
  rc = geozl_2d_decompress_c(NULL, 5, back, sizeof back, &n, 1, err,
                             sizeof err);
  REFUSED(rc, err, "frame is NULL");
  CHECK(n == 0);
  rc = geozl_2d_decompress_c(frame, sizeof frame, NULL, 5, &n, 1, err,
                             sizeof err);
  REFUSED(rc, err, "dst is NULL");
  CHECK(geozl_2d_frame_dsize_c(NULL, 16, &n) == ZL_ErrorCode_parameter_invalid);
  CHECK(geozl_2d_frame_coeffs_c(NULL, 16, NULL, 0, &n) ==
        ZL_ErrorCode_parameter_invalid);

  // A lossy open fits src, reading its samples before any compress.
  g = (geozl_2d_graph *)any;
  rc = geozl_2d_graph_open_c(&g, "id>zstd", W, 1, "LINEAR:MAX_ERROR=2",
                             GEOZL_DT_U16, GEOZL_NODATA_NONE, 0,
                             (const char *)src + 1, 16, 2, err, sizeof err);
  REFUSED(rc, err, "not aligned");
  CHECK(g == NULL);

  // A byte frame needs no alignment.
  size_t size = 0;
  CHECK(geozl_2d_compress_c("id>zstd", W, 1, NULL, GEOZL_DT_U8,
                            GEOZL_NODATA_NONE, 0, src, 64, 1, frame,
                            sizeof frame, &size, err, sizeof err) == 0);
  CHECK(geozl_2d_decompress_c(frame, size, (char *)any + 1, 64, &n, 1, err,
                              sizeof err) == 0);
  CHECK(n == 64);
}

// (NULL, 0) is an empty tile, the same frame as any other pointer with 0.
static void an_empty_tile_needs_no_pointer(void) {
  static unsigned char a[256], b[256];
  char err[256];
  const char *recipes[2] = {NULL, "LOG:MAX_ERROR=1%"};
  for (int k = 0; k < 2; ++k) {
    size_t na = 99, nb = 99;
    const int ra = geozl_2d_compress_c("planar>zigzag>pfor", 4, 1, recipes[k],
                                       GEOZL_DT_U16, GEOZL_NODATA_NONE, 0,
                                       NULL, 0, 2, a, sizeof a, &na, err,
                                       sizeof err);
    const int rb = geozl_2d_compress_c("planar>zigzag>pfor", 4, 1, recipes[k],
                                       GEOZL_DT_U16, GEOZL_NODATA_NONE, 0, any,
                                       0, 2, b, sizeof b, &nb, err, sizeof err);
    CHECK(ra == 0 && rb == 0);
    CHECK(na == nb && memcmp(a, b, na) == 0);
    size_t got = 99;
    CHECK(geozl_2d_decompress_c(a, na, NULL, 0, &got, 1, err, sizeof err) ==
              0 &&
          got == 0);
  }

  geozl_2d_graph *g = NULL;
  CHECK(geozl_2d_graph_open_c(&g, "planar>zigzag>pfor", 4, 1, NULL,
                              GEOZL_DT_U16, GEOZL_NODATA_NONE, 0, NULL, 0, 2,
                              err, sizeof err) == 0);
  size_t n = 99;
  CHECK(geozl_2d_compress_graph_c(g, NULL, 0, a, sizeof a, &n, err,
                                  sizeof err) == 0 &&
        n != 0);
  geozl_2d_graph_close_c(g);
  size_t comp = 0;
  double enc = 0, dec = 0;
  CHECK(geozl_2d_bench_c("id>zstd", 4, 1, NULL, GEOZL_DT_U16,
                         GEOZL_NODATA_NONE, 0, NULL, 0, 2, 1, 1, 1, &comp,
                         &enc, &dec, err, sizeof err) == 0);

  // OpenZL decodes a NULL dst into a buffer of its own and drops it.
  static unsigned char frame[4096];
  CHECK(geozl_2d_compress_c("id>zstd", W, 1, NULL, GEOZL_DT_U16,
                            GEOZL_NODATA_NONE, 0, src, 64, 2, frame,
                            sizeof frame, &n, err, sizeof err) == 0);
  size_t got = 99;
  CHECK(geozl_2d_decompress_c(frame, n, NULL, 0, &got, 1, err, sizeof err) ==
            ZL_ErrorCode_dstCapacity_tooSmall &&
        got == 0);
}

// Every 2D predictor fails on a partial row, and before OpenZL it says so
// instead of naming a node.
static void a_partial_row_is_refused(void) {
  static char names[128 * GEOZL_2D_NAME_MAX];
  static unsigned char frame[4096];
  char err[256];
  size_t count = 0;
  CHECK(geozl_2d_grid_c(NULL, 2, names, GEOZL_2D_NAME_MAX, 128, &count) == 0);
  for (size_t i = 0; i < count && i < 128; ++i) {
    const char *name = names + i * GEOZL_2D_NAME_MAX;
    const int rows = strncmp(name, "id>", 3) != 0 &&
                     strncmp(name, "delta_1d>", 9) != 0;
    geozl_2d_graph *g = NULL;
    CHECK(geozl_2d_graph_open_c(&g, name, 7, 1, NULL, GEOZL_DT_U16,
                                GEOZL_NODATA_NONE, 0, src, 98, 2, err,
                                sizeof err) == 0);
    size_t n = 99;
    const int rc = geozl_2d_compress_graph_c(g, src, 100, frame, sizeof frame,
                                             &n, err, sizeof err);
    if (rows) {
      CHECK(rc == ZL_ErrorCode_parameter_invalid && n == 0 &&
            strcmp(err, "100 samples do not divide into rows of 7") == 0);
    } else {
      CHECK(rc == 0);
    }
    if (rows ? rc != ZL_ErrorCode_parameter_invalid : rc != 0)
      printf("  %s: %d \"%s\"\n", name, rc, err);
    geozl_2d_graph_close_c(g);
  }
  // 0 and a width past the tile are one row.
  const uint32_t widths[3] = {0, 1000, 10};
  for (int k = 0; k < 3; ++k) {
    size_t n = 0;
    CHECK(geozl_2d_compress_c("planar>zigzag>pfor", widths[k], 1, NULL,
                              GEOZL_DT_U16, GEOZL_NODATA_NONE, 0, src, 100, 2,
                              frame, sizeof frame, &n, err, sizeof err) == 0);
  }
}

static void one_line(int rc, const char *err) {
  const char *code = ZL_ErrorCode_toString((ZL_ErrorCode)rc);
  CHECK(strncmp(err, code, strlen(code)) == 0);
  CHECK(strpbrk(err, "\n\t%") == NULL);
  CHECK(strstr(err, "Stack Trace") == NULL);
  CHECK(strstr(err, "Transform ID") == NULL);
  CHECK(strstr(err, "openzl/src") == NULL);
  if (strncmp(err, code, strlen(code)) != 0 || strpbrk(err, "\n\t%") != NULL)
    printf("  rc %d: \"%s\"\n", rc, err);
}

// OpenZL's context holds a stack trace with build paths, but a message is the
// code, the reason and the node.
static void openzl_messages_are_one_line(void) {
  static unsigned char frame[1 << 15], bad[1 << 15];
  char err[256];
  size_t n = 0;
  CHECK(geozl_2d_compress_c("planar>zigzag>pfor", 64, 1, NULL, GEOZL_DT_U16,
                            GEOZL_NODATA_NONE, 0, src, 64 * 64, 2, frame,
                            sizeof frame, &n, err, sizeof err) == 0);
  size_t got = 99;
  int rc = geozl_2d_decompress_c(frame, n - 3, back, sizeof back, &got, 1, err,
                                 sizeof err);
  CHECK(rc == ZL_ErrorCode_srcSize_tooSmall && got == 0);
  one_line(rc, err);
  rc = geozl_2d_decompress_c(frame, n, back, 10, &got, 1, err, sizeof err);
  CHECK(rc == ZL_ErrorCode_dstCapacity_tooSmall);
  one_line(rc, err);
  const unsigned char junk[64] = {1, 2, 3, 4};
  rc = geozl_2d_decompress_c(junk, sizeof junk, back, sizeof back, &got, 1, err,
                             sizeof err);
  CHECK(rc != 0);
  one_line(rc, err);
  size_t small = 99;
  rc = geozl_2d_compress_c("id>entropy", W, 1, NULL, GEOZL_DT_U16,
                           GEOZL_NODATA_NONE, 0, src, N, 2, bad, 3, &small, err,
                           sizeof err);
  CHECK(rc == ZL_ErrorCode_dstCapacity_tooSmall && small == 0);
  one_line(rc, err);

  // Damaged headers reach the codecs, whose refusals name their node.
  int named = 0;
  for (size_t at = 0; at < 64; ++at) {
    for (int bit = 0; bit < 8; bit += 2) {
      memcpy(bad, frame, n);
      bad[at] ^= (unsigned char)(1u << bit);
      rc = geozl_2d_decompress_c(bad, n, back, sizeof back, &got, 0, err,
                                 sizeof err);
      if (rc == 0)
        continue;
      one_line(rc, err);
      named |= strstr(err, "(node geozl.lossless.") != NULL;
    }
  }
  CHECK(named);

  // GeoZL's own reasons are no longer cut at 191 bytes.
  char recipe[300], wide[512];
  memset(recipe, 'x', sizeof recipe - 1);
  memcpy(recipe, "CUBE:", 5);
  recipe[sizeof recipe - 1] = '\0';
  rc = geozl_2d_compress_c("id>zstd", W, 1, recipe, GEOZL_DT_U16,
                           GEOZL_NODATA_NONE, 0, src, N, 2, frame, sizeof frame,
                           &n, wide, sizeof wide);
  CHECK(rc == ZL_ErrorCode_parameter_invalid && strlen(wide) > 300 &&
        strstr(wide, "each followed by a colon") != NULL);
}

// Every name fits a GEOZL_2D_NAME_MAX slot with room to spare, so none was cut,
// and a shorter slot is refused rather than cut.
static void grid_names_fit_their_slots(void) {
  static char names[128 * GEOZL_2D_NAME_MAX];
  char err[256];
  for (size_t w = 1; w <= 8; w *= 2) {
    size_t count = 0;
    CHECK(geozl_2d_grid_c(NULL, w, names, GEOZL_2D_NAME_MAX, 128, &count) ==
          0);
    CHECK(count > 0 && count <= 128);
    for (size_t i = 0; i < count && i < 128; ++i) {
      const char *name = names + i * GEOZL_2D_NAME_MAX;
      CHECK(strlen(name) + 1 < GEOZL_2D_NAME_MAX);
      geozl_2d_graph *g = NULL;
      const int rc = geozl_2d_graph_open_c(&g, name, W, 1, NULL,
                                           dtype_of_width[w], GEOZL_NODATA_NONE,
                                           0, any, N, w, err, sizeof err);
      CHECK(rc == 0);
      if (rc != 0)
        printf("  %s: %s\n", name, err);
      geozl_2d_graph_close_c(g);
    }
  }
  size_t count = 7;
  CHECK(geozl_2d_grid_c(NULL, 2, names, GEOZL_2D_NAME_MAX - 1, 128, &count) ==
            -1 &&
        count == 7);
  CHECK(geozl_2d_grid_c(NULL, 2, NULL, GEOZL_2D_NAME_MAX, 128, &count) == -1);
  CHECK(geozl_2d_grid_c("planar", 2, NULL, 0, 0, &count) == 0 && count > 0);
}

int main(void) {
  fill();
  node_quant_matches_the_2d_path("LINEAR:MAX_ERROR=2", 2.0);
  node_quant_matches_the_2d_path("LOG:MAX_ERROR=1%", 1.0);
  node_quant_matches_the_2d_path("SQRT:MAX_ERROR=1N", INFINITY);
  node_quant_matches_the_2d_path("SQRT:MAX_ERROR=1N,A=4,B=1", INFINITY);
  node_quant_refuses(NULL, GEOZL_DT_U16, "lossless");
  node_quant_refuses("", GEOZL_DT_U16, "lossless");
  node_quant_refuses("CUBE:MAX_ERROR=1", GEOZL_DT_U16, "families");
  node_quant_refuses("LINEAR:MAX_ERROR=-1", GEOZL_DT_U16, "positive");
  node_quant_refuses("LINEAR:MAX_ERROR=1", 99, "dtype 99");
  quant_builders_refuse_bad_input();
  builders_refuse_a_null_compressor();
  quant_builders_take_valid_parameters();
  frame_dsize_tells_empty_from_unreadable();
  compress_bound_covers_noise();
  nodata_modes_need_their_dtype();
  null_arguments_are_refused();
  an_empty_tile_needs_no_pointer();
  a_partial_row_is_refused();
  openzl_messages_are_one_line();
  grid_names_fit_their_slots();
  if (failures != 0) {
    printf("test_c_api: %d failures\n", failures);
    return 1;
  }
  printf("test_c_api: ok\n");
  return 0;
}
