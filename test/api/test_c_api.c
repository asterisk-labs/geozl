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

// The frame a hand built quantizer -> entropy graph writes.
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
  frame_dsize_tells_empty_from_unreadable();
  compress_bound_covers_noise();
  if (failures != 0) {
    printf("test_c_api: %d failures\n", failures);
    return 1;
  }
  printf("test_c_api: ok\n");
  return 0;
}
