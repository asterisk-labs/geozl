#include "geozl/dtype.h"
#include "lossy/lossy_recipe.h"

#include <math.h>
#include <stdint.h>
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

static float raster[N];

// A ramp with noise scaled the way a sensor scales it, so the blind fit has a
// curve to find. 256 on a side because the fit wants a few hundred blocks.
static void build_raster(void) {
  unsigned s = 1;
  for (int i = 0; i < N; ++i) {
    const double mu = 50.0 + 3950.0 * ((double)i / N);
    double u = 0.0;
    // Four uniforms sum to variance 1/3, so sqrt(3V) scales them to variance V.
    for (int k = 0; k < 4; ++k)
      u += (double)((s = s * 1103515245u + 12345u) >> 16 & 0xFFFF) / 65535.0 - 0.5;
    raster[i] = (float)(mu + u * sqrt(3.0 * (100.0 + mu)));
  }
}

static geozl_lossy_family parse_family(const char *s) {
  geozl_lossy_recipe r;
  char err[256];
  if (geozl_lossy_parse(s, &r, err, sizeof(err)) != 0)
    return GEOZL_LOSSY_NONE;
  return r.family;
}

static void test_routing(void) {
  CHECK(parse_family(NULL) == GEOZL_LOSSY_NONE);
  CHECK(parse_family("") == GEOZL_LOSSY_NONE);
  CHECK(parse_family("LINEAR:MAX_ERROR=0.5") == GEOZL_LOSSY_LINEAR);
  CHECK(parse_family("LOG:MAX_ERROR=1%") == GEOZL_LOSSY_LOG);
  CHECK(parse_family("SQRT:MAX_ERROR=0.5N") == GEOZL_LOSSY_SQRT);

  // The old grammar and a family without its colon are both refused.
  geozl_lossy_recipe r;
  char err[256];
  CHECK(geozl_lossy_parse("abs:0.5", &r, err, sizeof(err)) != 0);
  CHECK(geozl_lossy_parse("LINEAR", &r, err, sizeof(err)) != 0);
  CHECK(geozl_lossy_parse("linear:MAX_ERROR=1", &r, err, sizeof(err)) != 0);

  // A refused recipe reads as lossless, so ignoring the return code is safe.
  CHECK(r.family == GEOZL_LOSSY_NONE);
}

static void test_fit_geometry(void) {
  geozl_lossy_recipe r, before;
  char err[256];

  // Not a rectangle, so there is nothing to measure blocks on.
  CHECK(geozl_lossy_parse("SQRT:MAX_ERROR=0.5N", &r, err, sizeof(err)) == 0);
  CHECK(geozl_lossy_fit(&r, raster, GEOZL_DT_F32, 0, N, err, sizeof(err)) != 0);
  CHECK(geozl_lossy_fit(&r, raster, GEOZL_DT_F32, 7, N, err, sizeof(err)) != 0);
  CHECK(r.as.sqrt.have_ab == 0);

  // A no-op for every family that is not a curveless SQRT.
  const char *quiet[] = {"", "LINEAR:MAX_ERROR=0.5", "LOG:MAX_ERROR=1%",
                         "SQRT:MAX_ERROR=0.5N,A=100,B=1"};
  for (unsigned i = 0; i < 4; ++i) {
    CHECK(geozl_lossy_parse(quiet[i], &r, err, sizeof(err)) == 0);
    before = r;
    CHECK(geozl_lossy_fit(&r, raster, GEOZL_DT_F32, SIDE, N, err,
                          sizeof(err)) == 0);
    CHECK(memcmp(&r, &before, sizeof(r)) == 0);
  }
}

// The one branch the fuzzer cannot reach, since the fit wants a raster larger
// than a fuzz input.
static void test_blind_fit(void) {
  geozl_lossy_recipe r;
  char err[256] = {0};
  CHECK(geozl_lossy_parse("SQRT:MAX_ERROR=0.5N", &r, err, sizeof(err)) == 0);
  if (geozl_lossy_fit(&r, raster, GEOZL_DT_F32, SIDE, N, err, sizeof(err)) !=
      0) {
    printf("  FAIL the fit found no curve: %s\n", err);
    ++failures;
    return;
  }
  CHECK(r.as.sqrt.have_ab == 1);
  CHECK(r.as.sqrt.b > 0.5 && r.as.sqrt.b < 2.0); // the raster was drawn at 1.0
  CHECK(isfinite(r.as.sqrt.a) && r.as.sqrt.a >= 0.0);

  // A recipe that already carries a curve is left alone.
  geozl_lossy_recipe again = r;
  CHECK(geozl_lossy_fit(&again, raster, GEOZL_DT_F32, SIDE, N, err,
                        sizeof(err)) == 0);
  CHECK(memcmp(&again, &r, sizeof(r)) == 0);
}

static void test_resolve(void) {
  static const char *recipes[] = {"", "LINEAR:MAX_ERROR=0.5", "LOG:MAX_ERROR=1%",
                                  "SQRT:MAX_ERROR=0.5N,A=100,B=1"};
  for (unsigned i = 0; i < 4; ++i) {
    geozl_lossy_recipe r;
    geozl_lossy_plan p;
    char err[256] = {0};
    CHECK(geozl_lossy_parse(recipes[i], &r, err, sizeof(err)) == 0);
    if (geozl_lossy_resolve(&r, raster, GEOZL_DT_F32, N, &p, err,
                            sizeof(err)) != 0) {
      printf("  FAIL resolve refused \"%s\": %s\n", recipes[i], err);
      ++failures;
      continue;
    }
    CHECK(p.family == r.family);
    if (p.family == GEOZL_LOSSY_NONE)
      continue;
    const double step = p.family == GEOZL_LOSSY_LINEAR ? p.as.linear.step
                        : p.family == GEOZL_LOSSY_LOG  ? p.as.log.step
                                                       : p.as.sqrt.step;
    CHECK(isfinite(step) && step > 0.0);
  }

  // A SQRT recipe that carries no curve and was never fitted has nothing to cut
  // a grid from, and says so rather than guessing.
  geozl_lossy_recipe r;
  geozl_lossy_plan p;
  char err[256] = {0};
  CHECK(geozl_lossy_parse("SQRT:MAX_ERROR=0.5N", &r, err, sizeof(err)) == 0);
  CHECK(geozl_lossy_resolve(&r, raster, GEOZL_DT_F32, N, &p, err,
                            sizeof(err)) != 0);
}

static int plan_for(const char *text, const void *src, int dtype, size_t n,
                    geozl_lossy_plan *plan, char *err, size_t errSize) {
  geozl_lossy_recipe recipe;
  return geozl_lossy_parse(text, &recipe, err, errSize) ||
         geozl_lossy_resolve(&recipe, src, dtype, n, plan, err, errSize);
}

static void test_reuse_domain(void) {
  char err[256] = {0};
  geozl_lossy_plan plan;

  const float base[] = {1.0f, 25.0f, 100.0f};
  const float inside[] = {2.0f, 50.0f, 90.0f};
  const float negative[] = {-1.0f, 25.0f};
  const float wider[] = {1.0f, 120.0f};

  CHECK(plan_for("LINEAR:MAX_ERROR=1", base, GEOZL_DT_F32, 3, &plan,
                 err, sizeof(err)) == 0);
  CHECK(geozl_lossy_check_domain(&plan, inside, GEOZL_DT_F32, 3, 0, 0,
                                 err, sizeof(err)) == 0);
  CHECK(geozl_lossy_check_domain(&plan, negative, GEOZL_DT_F32, 2, 0, 0,
                                 err, sizeof(err)) != 0);
  CHECK(strstr(err, "negative samples") != NULL);
  CHECK(geozl_lossy_check_domain(&plan, wider, GEOZL_DT_F32, 2, 0, 0,
                                 err, sizeof(err)) != 0);
  CHECK(strstr(err, "magnitudes up to") != NULL);

  const float sentinel = -9999.0f;
  const float withHole[] = {sentinel, 50.0f};
  const float withHoleAndOutlier[] = {sentinel, 120.0f};
  uint32_t sentinelBits;
  memcpy(&sentinelBits, &sentinel, sizeof(sentinelBits));
  CHECK(geozl_lossy_check_domain(&plan, withHole, GEOZL_DT_F32, 2, 1,
                                 sentinelBits, err, sizeof(err)) == 0);
  CHECK(geozl_lossy_check_domain(&plan, withHoleAndOutlier, GEOZL_DT_F32,
                                 2, 1, sentinelBits, err,
                                 sizeof(err)) != 0);

  // LOG's index grid spans the floating-point dtype, but its non-negative flag
  // still makes a later negative tile unsafe.
  CHECK(plan_for("LOG:MAX_ERROR=1%", base, GEOZL_DT_F32, 3, &plan, err,
                 sizeof(err)) == 0);
  CHECK(geozl_lossy_check_domain(&plan, wider, GEOZL_DT_F32, 2, 0, 0,
                                 err, sizeof(err)) == 0);
  CHECK(geozl_lossy_check_domain(&plan, negative, GEOZL_DT_F32, 2, 0, 0,
                                 err, sizeof(err)) != 0);

  CHECK(plan_for("SQRT:MAX_ERROR=0.5N,A=100,B=1", base, GEOZL_DT_F32,
                 3, &plan, err, sizeof(err)) == 0);
  CHECK(geozl_lossy_check_domain(&plan, inside, GEOZL_DT_F32, 3, 0, 0,
                                 err, sizeof(err)) == 0);
  CHECK(geozl_lossy_check_domain(&plan, wider, GEOZL_DT_F32, 2, 0, 0,
                                 err, sizeof(err)) != 0);
  CHECK(strstr(err, "values from") != NULL);

  // Integer LINEAR stores reconstructed values and spans the dtype. Only a
  // signedness change relative to the opening raster needs a guard.
  const int16_t ibase[] = {1, 2, 3};
  const int16_t ibig[] = {30000};
  CHECK(plan_for("LINEAR:MAX_ERROR=2", ibase, GEOZL_DT_I16, 3, &plan,
                 err, sizeof(err)) == 0);
  CHECK(geozl_lossy_check_domain(&plan, ibig, GEOZL_DT_I16, 1, 0, 0,
                                 err, sizeof(err)) == 0);
}

int main(void) {
  build_raster();
  test_routing();
  test_fit_geometry();
  test_blind_fit();
  test_resolve();
  test_reuse_domain();

  if (failures) {
    printf("test_lossy_recipe: %d failed\n", failures);
    return 1;
  }
  printf("test_lossy_recipe: ok\n");
  return 0;
}
