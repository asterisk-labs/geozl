// The parts of geozl/geozl.h a C caller reaches without the 2d entry points,
// which need OpenZL linked in and so cannot run from the Python tests.

#include "geozl/dtype.h"
#include "geozl/geozl.h"

#include "openzl/zl_compress.h"
#include "openzl/zl_compressor.h"
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
  if (failures != 0) {
    printf("test_c_api: %d failures\n", failures);
    return 1;
  }
  printf("test_c_api: ok\n");
  return 0;
}
