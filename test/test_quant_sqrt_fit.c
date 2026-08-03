#include "geozl/dtype.h"
#include "quant_sqrt/quant_sqrt_fit.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);                 \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

#define SIDE 256
#define N (SIDE * SIDE)
#define TRUE_A 100.0
#define TRUE_B 1.0

static float wide[N], dark[N], bright[N], flat[N];

static unsigned rng = 1;

// Four uniforms sum to variance 1/3, so sqrt(3V) scales them to variance V.
static double noise(double var) {
  double u = 0.0;
  for (int k = 0; k < 4; ++k)
    u += (double)((rng = rng * 1103515245u + 12345u) >> 16 & 0xFFFF) / 65535.0 -
         0.5;
  return u * sqrt(3.0 * var);
}

static void draw(float *dst, double lo, double hi) {
  for (int i = 0; i < N; ++i) {
    const double mu = lo + (hi - lo) * ((double)i / N);
    dst[i] = (float)(mu + noise(TRUE_A + TRUE_B * mu));
  }
}

static void build(void) {
  draw(wide, 50.0, 4000.0);
  draw(dark, 50.0, 300.0);
  draw(bright, 3000.0, 4000.0);
  for (int i = 0; i < N; ++i)
    flat[i] = 1.0f;
}

static void test_recovers_the_curve(void) {
  quant_sqrt_noise n;
  char err[256] = {0};
  if (quant_sqrt_fit(wide, GEOZL_DT_F32, SIDE, SIDE, &n, err, sizeof(err)) !=
      0) {
    printf("  FAIL no curve on a raster that has one: %s\n", err);
    ++failures;
    return;
  }
  CHECK(n.ok == 1);
  CHECK(n.b > 0.5 * TRUE_B && n.b < 2.0 * TRUE_B);
  CHECK(n.a >= 0.0 && isfinite(n.a));
  CHECK(n.blocks > 0 && n.bins > 0);
  CHECK(n.range > 1.0);
}

static void test_refusals(void) {
  quant_sqrt_noise n;
  char err[256];

  // No dynamic range, so nothing separates a from b.
  CHECK(quant_sqrt_fit(flat, GEOZL_DT_F32, SIDE, SIDE, &n, err, sizeof(err)) !=
        0);

  // Too small to hold blocks.
  CHECK(quant_sqrt_fit(wide, GEOZL_DT_F32, 4, 4, &n, err, sizeof(err)) != 0);
  CHECK(quant_sqrt_fit(wide, GEOZL_DT_F32, 0, 0, &n, err, sizeof(err)) != 0);

  // A refused fit leaves ok clear, whatever else it wrote.
  CHECK(n.ok == 0);
}

// The reason the accumulator exists. One end of the range on its own gives a
// plausible curve that is not the sensor's, and only colin says so.
static void test_pooling_beats_its_parts(void) {
  quant_sqrt_noise alone, pooled;
  char err[256] = {0};
  CHECK(quant_sqrt_fit(bright, GEOZL_DT_F32, SIDE, SIDE, &alone, err,
                       sizeof(err)) == 0);

  quant_sqrt_accum acc;
  quant_sqrt_accum_init(&acc);
  CHECK(quant_sqrt_accum_push(&acc, dark, GEOZL_DT_F32, SIDE, SIDE) == 0);
  CHECK(quant_sqrt_accum_push(&acc, bright, GEOZL_DT_F32, SIDE, SIDE) == 0);
  const int rc = quant_sqrt_accum_solve(&acc, &pooled, err, sizeof(err));
  quant_sqrt_accum_free(&acc);

  if (rc != 0) {
    printf("  FAIL pooling two halves found no curve: %s\n", err);
    ++failures;
    return;
  }
  CHECK(pooled.colin < alone.colin);
  CHECK(fabs(pooled.b - TRUE_B) < fabs(alone.b - TRUE_B));
}

static void test_accum_matches_fit_on_one_raster(void) {
  quant_sqrt_noise one, viaAcc;
  char err[256] = {0};
  CHECK(quant_sqrt_fit(wide, GEOZL_DT_F32, SIDE, SIDE, &one, err,
                       sizeof(err)) == 0);

  quant_sqrt_accum acc;
  quant_sqrt_accum_init(&acc);
  CHECK(quant_sqrt_accum_push(&acc, wide, GEOZL_DT_F32, SIDE, SIDE) == 0);
  CHECK(quant_sqrt_accum_solve(&acc, &viaAcc, err, sizeof(err)) == 0);
  quant_sqrt_accum_free(&acc);

  CHECK(one.a == viaAcc.a && one.b == viaAcc.b);
  CHECK(one.blocks == viaAcc.blocks);
}

static void test_free_of_an_empty_accum(void) {
  quant_sqrt_accum acc;
  quant_sqrt_noise n;
  char err[256];
  quant_sqrt_accum_init(&acc);
  CHECK(quant_sqrt_accum_solve(&acc, &n, err, sizeof(err)) != 0);
  quant_sqrt_accum_free(&acc);
  quant_sqrt_accum_free(&acc); // twice, since a caller unwinding may do that
}

int main(void) {
  build();
  test_recovers_the_curve();
  test_refusals();
  test_pooling_beats_its_parts();
  test_accum_matches_fit_on_one_raster();
  test_free_of_an_empty_accum();

  if (failures) {
    printf("test_quant_sqrt_fit: %d failed\n", failures);
    return 1;
  }
  printf("test_quant_sqrt_fit: ok\n");
  return 0;
}