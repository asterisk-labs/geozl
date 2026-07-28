#include "quant/decode_quant_kernel.h"
#include "quant/encode_quant_kernel.h"
#include "quant/quant_spec.h"

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

#define N 4096

static float f32src[N], f32back[N];
static int32_t idx32[N];
static double f64src[N], f64back[N];
static int64_t idx64[N];
static uint16_t u16src[N], u16back[N];

static uint32_t rng_state = 0x9E3779B9u;

static double urand(void) {
  rng_state = rng_state * 1664525u + 1013904223u;
  return (double)(rng_state >> 8) / 16777216.0;
}

// Run one spec over one f32 tile and check the bound it declares.
static void f32_case(const char *spec, const float *src, size_t n) {
  char err[256] = {0};
  quant_spec sp;
  CHECK(quant_spec_parse(spec, &sp, err, sizeof(err)) == 0);

  double lo, hi;
  int neg;
  CHECK(quant_scan(src, Q_F32, n, &lo, &hi, &neg) == 0);

  quant_params p;
  if (quant_spec_resolve(&sp, Q_F32, lo, hi, neg, &p, err, sizeof(err)) != 0) {
    printf("  FAIL resolve %s: %s\n", spec, err);
    ++failures;
    return;
  }

  CHECK(quant_encode(idx32, src, &p, Q_F32, n) == 0);
  CHECK(quant_decode(f32back, idx32, &p, Q_F32, n) == 0);

  double worst = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double x = (double)src[i];
    const double e = fabs(x - (double)f32back[i]);
    double bound;
    if (sp.curve == QUANT_CURVE_LINEAR)
      bound = sp.abs_err;
    else if (sp.curve == QUANT_CURVE_LOG)
      bound = sp.rel_err * fabs(x);
    else
      bound = sp.shot_k * sqrt(sp.shot_a + sp.shot_b * x);
    // A relative bound is also met when the two values are identical, which is
    // what preserves zero and the subnormals a geometric grid cannot address.
    const double r = bound > 0.0 ? e / bound : (e == 0.0 ? 0.0 : INFINITY);
    if (x == (double)f32back[i])
      continue;
    if (r > worst)
      worst = r;
  }
  if (worst > 1.0) {
    printf("  FAIL %s: worst error is %.6f of the declared bound\n", spec,
           worst);
    ++failures;
  }
}

int main(void) {
  printf("test_quant\n");

  // A smooth field over four decades, the shape the sqrt and log curves are
  // meant for.
  for (size_t i = 0; i < N; ++i) {
    const double t = (double)i / (double)N;
    f32src[i] = (float)(1e-3 * pow(10.0, 4.0 * t) * (0.5 + urand()));
    f64src[i] = (double)f32src[i];
  }

  f32_case("abs:0.5", f32src, N);
  f32_case("rel:1%", f32src, N);
  f32_case("rel:0.17%", f32src, N);
  f32_case("rel:10.71%", f32src, N);
  f32_case("shot:a=4,b=0.5,k=0.5", f32src, N);

  // Zero and the subnormals a relative grid cannot address have to come back
  // exactly, which is the second clause of the bound and the case that trips
  // mantissa rounding schemes.
  for (size_t i = 0; i < 64; ++i)
    f32src[i] = 0.0f;
  for (size_t i = 64; i < 128; ++i)
    f32src[i] = (float)((double)(i - 63) * 1.4012984643e-45);
  f32_case("rel:1.1%", f32src, N);
  f32_case("rel:10.71%", f32src, N);
  {
    char err[256];
    quant_spec sp;
    quant_params p;
    double lo, hi;
    int neg;
    CHECK(quant_spec_parse("rel:10.71%", &sp, err, sizeof(err)) == 0);
    CHECK(quant_scan(f32src, Q_F32, N, &lo, &hi, &neg) == 0);
    CHECK(quant_spec_resolve(&sp, Q_F32, lo, hi, neg, &p, err, sizeof(err)) ==
          0);
    CHECK(p.nsub != 0);
    CHECK(quant_encode(idx32, f32src, &p, Q_F32, N) == 0);
    CHECK(quant_decode(f32back, idx32, &p, Q_F32, N) == 0);
    for (size_t i = 0; i < 128; ++i)
      CHECK(f32back[i] == f32src[i]);
  }

  // f64 round trip on the same shape.
  {
    char err[256];
    quant_spec sp;
    quant_params p;
    double lo, hi;
    int neg;
    for (size_t i = 0; i < N; ++i)
      f64src[i] = 1e-9 * pow(10.0, 7.0 * (double)i / (double)N);
    CHECK(quant_spec_parse("rel:0.94%", &sp, err, sizeof(err)) == 0);
    CHECK(quant_scan(f64src, Q_F64, N, &lo, &hi, &neg) == 0);
    CHECK(quant_spec_resolve(&sp, Q_F64, lo, hi, neg, &p, err, sizeof(err)) ==
          0);
    CHECK(quant_encode(idx64, f64src, &p, Q_F64, N) == 0);
    CHECK(quant_decode(f64back, idx64, &p, Q_F64, N) == 0);
    for (size_t i = 0; i < N; ++i)
      CHECK(fabs(f64src[i] - f64back[i]) <= 0.0094 * fabs(f64src[i]));
  }

  // The linear curve on integers is the path the codec always had: the stream
  // holds the reconstruction and the decoder copies.
  {
    char err[256];
    quant_spec sp;
    quant_params p;
    for (size_t i = 0; i < N; ++i)
      u16src[i] = (uint16_t)(urand() * 60000.0);
    CHECK(quant_spec_parse("abs:8", &sp, err, sizeof(err)) == 0);
    CHECK(quant_spec_resolve(&sp, Q_U16, 1.0, 60000.0, 0, &p, err,
                             sizeof(err)) == 0);
    CHECK((p.flags & QUANT_FLAG_STORE_VALUES) != 0);
    CHECK(quant_encode(u16back, u16src, &p, Q_U16, N) == 0);
    uint16_t out[N];
    CHECK(quant_decode(out, u16back, &p, Q_U16, N) == 0);
    for (size_t i = 0; i < N; ++i)
      CHECK(abs((int)out[i] - (int)u16src[i]) <= 8);
  }

  // Lossless is a step of zero and copies through.
  {
    char err[256];
    quant_spec sp;
    quant_params p;
    CHECK(quant_spec_parse(NULL, &sp, err, sizeof(err)) == 0);
    CHECK(sp.mode == QUANT_SPEC_LOSSLESS);
    CHECK(quant_spec_resolve(&sp, Q_F32, 1.0, 2.0, 0, &p, err, sizeof(err)) ==
          0);
    CHECK(p.step == 0.0);
    CHECK(quant_encode(idx32, f32src, &p, Q_F32, N) == 0);
    CHECK(quant_decode(f32back, idx32, &p, Q_F32, N) == 0);
    CHECK(memcmp(f32src, f32back, sizeof(float) * N) == 0);
  }

  // Rejections. A bare rel without the percent sign is the typo that would
  // otherwise quantize a hundred times coarser than intended.
  {
    char err[256];
    quant_spec sp;
    CHECK(quant_spec_parse("rel:1", &sp, err, sizeof(err)) != 0);
    CHECK(quant_spec_parse("abs:0", &sp, err, sizeof(err)) != 0);
    CHECK(quant_spec_parse("abs:-1", &sp, err, sizeof(err)) != 0);
    CHECK(quant_spec_parse("abs:nan", &sp, err, sizeof(err)) != 0);
    CHECK(quant_spec_parse("rel:120%", &sp, err, sizeof(err)) != 0);
    CHECK(quant_spec_parse("shot:b=1,a=1,k=1", &sp, err, sizeof(err)) != 0);
    CHECK(quant_spec_parse("nope:1", &sp, err, sizeof(err)) != 0);

    quant_params p;
    CHECK(quant_spec_parse("shot:a=0,b=1,k=0.5", &sp, err, sizeof(err)) == 0);
    CHECK(quant_spec_resolve(&sp, Q_F32, 1.0, 2.0, 1, &p, err, sizeof(err)) !=
          0);
    // A bound far below what a 32-bit index can address has to be refused, not
    // saturated.
    CHECK(quant_spec_parse("rel:1e-7%", &sp, err, sizeof(err)) == 0);
    CHECK(quant_spec_resolve(&sp, Q_F32, 1e-30, 1e30, 0, &p, err,
                             sizeof(err)) != 0);
  }

  if (failures == 0)
    printf("  ok\n");
  return failures != 0;
}
