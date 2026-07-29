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

  // A relative bound is symmetric, so the sign has to survive the trip and the
  // magnitude has to hold on both sides of zero.
  {
    char err[256];
    quant_spec sp;
    quant_params p;
    double lo, hi;
    int neg;
    for (size_t i = 0; i < N; ++i) {
      const double m = pow(10.0, -6.0 + 12.0 * (double)i / (double)N);
      f32src[i] = (float)((i % 3 == 0) ? -m : m);
    }
    CHECK(quant_spec_parse("rel:1%", &sp, err, sizeof(err)) == 0);
    CHECK(quant_scan(f32src, Q_F32, N, &lo, &hi, &neg) == 0);
    CHECK(neg == 1);
    CHECK(quant_spec_resolve(&sp, Q_F32, lo, hi, neg, &p, err, sizeof(err)) ==
          0);
    CHECK(quant_encode(idx32, f32src, &p, Q_F32, N) == 0);
    CHECK(quant_decode(f32back, idx32, &p, Q_F32, N) == 0);
    for (size_t i = 0; i < N; ++i) {
      CHECK((f32src[i] < 0.0f) == (f32back[i] < 0.0f));
      CHECK(fabs((double)f32src[i] - (double)f32back[i]) <=
            0.01 * fabs((double)f32src[i]));
    }
  }

  // Wide enough that the index range leaves the reconstruction table and the
  // decoder falls back to evaluating the curve per sample. Both paths have to
  // agree, so this is the same check as above with a tighter bound.
  {
    char err[256];
    quant_spec sp;
    quant_params p;
    double lo, hi;
    int neg;
    for (size_t i = 0; i < N; ++i)
      f32src[i] = (float)pow(10.0, -6.0 + 12.0 * (double)i / (double)N);
    CHECK(quant_spec_parse("rel:0.01%", &sp, err, sizeof(err)) == 0);
    CHECK(quant_scan(f32src, Q_F32, N, &lo, &hi, &neg) == 0);
    CHECK(quant_spec_resolve(&sp, Q_F32, lo, hi, neg, &p, err, sizeof(err)) ==
          0);
    CHECK(fabs(quant_fwd((double)f32src[N - 1], &p)) > 8192.0);
    CHECK(quant_encode(idx32, f32src, &p, Q_F32, N) == 0);
    CHECK(quant_decode(f32back, idx32, &p, Q_F32, N) == 0);
    for (size_t i = 0; i < N; ++i)
      CHECK(fabs((double)f32src[i] - (double)f32back[i]) <=
            0.0001 * fabs((double)f32src[i]));
  }

  // Shot noise on an unsigned integer raster, which is how optical reflectance
  // actually arrives. The curve reconstructs below zero near the bottom of the
  // range, so this is also the case that pins down the clamp: without it the
  // negative reconstruction wraps and the sample comes back as garbage.
  {
    char err[256];
    quant_spec sp;
    quant_params p;
    double lo, hi;
    int neg;
    uint16_t out[N];
    for (size_t i = 0; i < N; ++i)
      u16src[i] = (uint16_t)(i * 5);
    CHECK(quant_spec_parse("shot:a=100,b=1,k=0.5", &sp, err, sizeof(err)) == 0);
    CHECK(quant_scan(u16src, Q_U16, N, &lo, &hi, &neg) == 0);
    CHECK(quant_spec_resolve(&sp, Q_U16, lo, hi, neg, &p, err, sizeof(err)) ==
          0);
    CHECK(quant_inv(quant_fwd(0.0, &p), &p) < 0.0);
    CHECK(quant_encode(u16back, u16src, &p, Q_U16, N) == 0);
    CHECK(quant_decode(out, u16back, &p, Q_U16, N) == 0);
    for (size_t i = 0; i < N; ++i) {
      const double x = (double)u16src[i];
      CHECK(fabs(x - (double)out[i]) <= 0.5 * sqrt(100.0 + x));
    }
  }

  // Encode and decode have to round a reconstruction the same way. Truncating
  // on one side and rounding on the other puts every integer reconstruction up
  // to a unit off, which no bound survives.
  {
    char err[256];
    quant_spec sp;
    quant_params p;
    uint16_t out[N];
    for (size_t i = 0; i < N; ++i)
      u16src[i] = (uint16_t)(i * 13);
    CHECK(quant_spec_parse("shot:a=100,b=1,k=0.5", &sp, err, sizeof(err)) == 0);
    CHECK(quant_spec_resolve(&sp, Q_U16, 13.0, 13.0 * (N - 1), 0, &p, err,
                             sizeof(err)) == 0);
    CHECK(quant_encode(u16back, u16src, &p, Q_U16, N) == 0);
    CHECK(quant_decode(out, u16back, &p, Q_U16, N) == 0);
    for (size_t i = 0; i < N; ++i) {
      const double lo = quant_value_lo(Q_U16), hi = quant_value_hi(Q_U16);
      const double want =
          QUANT_RT_INT(quant_inv((double)u16back[i], &p), lo, hi);
      CHECK((double)out[i] == want);
    }
  }

  // f16 steps further near the top of its range than most bounds allow, so the
  // rounding alone breaks the bound. Refused rather than declared and missed.
  {
    char err[256];
    quant_spec sp;
    quant_params p;
    CHECK(quant_spec_parse("abs:5", &sp, err, sizeof(err)) == 0);
    CHECK(quant_spec_resolve(&sp, Q_F16, 1.0, 60000.0, 0, &p, err,
                             sizeof(err)) != 0);
    CHECK(quant_spec_resolve(&sp, Q_F32, 1.0, 60000.0, 0, &p, err,
                             sizeof(err)) == 0);
    // and a linear grid finer than the index can address is refused too, which
    // the curve used to skip on its way out
    CHECK(quant_spec_parse("abs:0.05", &sp, err, sizeof(err)) == 0);
    CHECK(quant_spec_resolve(&sp, Q_F16, 1.0, 60000.0, 0, &p, err,
                             sizeof(err)) != 0);
  }

  // Every comparison against a NaN is false, so a pair of clamps lets one
  // through to a cast, and casting a NaN to an integer is undefined. It has to
  // land on an index, not on whatever the hardware felt like.
  {
    char err[256];
    quant_spec sp;
    quant_params p;
    for (size_t i = 0; i < N; ++i)
      f32src[i] = (float)(i + 1);
    f32src[0] = NAN;
    f32src[1] = -NAN;
    f32src[2] = INFINITY;
    f32src[3] = -INFINITY;
    double lo, hi;
    int neg;
    CHECK(quant_scan(f32src, Q_F32, N, &lo, &hi, &neg) == 0);
    CHECK(lo == 5.0); // the four are skipped, the first real sample is 5
    CHECK(quant_spec_parse("abs:0.5", &sp, err, sizeof(err)) == 0);
    CHECK(quant_spec_resolve(&sp, Q_F32, lo, hi, neg, &p, err, sizeof(err)) ==
          0);
    CHECK(quant_encode(idx32, f32src, &p, Q_F32, N) == 0);
    CHECK(idx32[0] == 0 && idx32[1] == 0);
    CHECK(quant_decode(f32back, idx32, &p, Q_F32, N) == 0);
    for (size_t i = 4; i < N; ++i)
      CHECK(fabs((double)f32src[i] - (double)f32back[i]) <= 0.5);
  }

  // A float reconstruction is rounded to its own width, and that rounding grows
  // with the magnitude, so a fixed bound stops being free once the data runs
  // large against it. At 1.6e10 a float32 already steps by 1024.
  {
    char err[256];
    quant_spec sp;
    quant_params p;
    for (size_t i = 0; i < N; ++i)
      f32src[i] = (float)(-1.6392371e10 + (double)i * 4096.0);
    double lo, hi;
    int neg;
    CHECK(quant_scan(f32src, Q_F32, N, &lo, &hi, &neg) == 0);
    CHECK(quant_spec_parse("abs:1000", &sp, err, sizeof(err)) == 0);
    CHECK(quant_spec_resolve(&sp, Q_F32, lo, hi, neg, &p, err, sizeof(err)) ==
          0);
    CHECK(p.step < 2000.0); // the rounding came out of the step
    CHECK(quant_encode(idx32, f32src, &p, Q_F32, N) == 0);
    CHECK(quant_decode(f32back, idx32, &p, Q_F32, N) == 0);
    for (size_t i = 0; i < N; ++i)
      CHECK(fabs((double)f32src[i] - (double)f32back[i]) <= 1000.0);
  }

  // The top of the log grid can sit above the largest finite value of the
  // output type, and letting it turn into an infinity there costs more than the
  // bound allows, besides being no answer at all.
  {
    char err[256];
    quant_spec sp;
    quant_params p;
    for (size_t i = 0; i < N; ++i)
      f32src[i] = (float)(3.0e38 * (0.5 + 0.5 * (double)i / (double)N));
    f32src[N - 1] = 3.3951636e38f;
    double lo, hi;
    int neg;
    CHECK(quant_scan(f32src, Q_F32, N, &lo, &hi, &neg) == 0);
    CHECK(quant_spec_parse("rel:1%", &sp, err, sizeof(err)) == 0);
    CHECK(quant_spec_resolve(&sp, Q_F32, lo, hi, neg, &p, err, sizeof(err)) ==
          0);
    CHECK(quant_encode(idx32, f32src, &p, Q_F32, N) == 0);
    CHECK(quant_decode(f32back, idx32, &p, Q_F32, N) == 0);
    for (size_t i = 0; i < N; ++i) {
      CHECK(isfinite(f32back[i]));
      CHECK(fabs((double)f32src[i] - (double)f32back[i]) <=
            0.01 * fabs((double)f32src[i]));
    }
  }

  // A step wide enough to leave the range of the integer it multiplies, and an
  // index stream whose extremes are further apart than a signed subtraction can
  // express. Both got past the header checks.
  {
    quant_params p;
    uint8_t u8out[N];
    memset(&p, 0, sizeof(p));
    p.curve = QUANT_CURVE_LINEAR;
    p.step = 8.5926701187887794e245;
    memset(u16src, 0x5A, sizeof(u16src));
    CHECK(quant_decode(u8out, u16src, &p, Q_U8, 64) == 0);

    p.curve = QUANT_CURVE_LOG;
    p.step = 0.02;
    p.offset = 1.0;
    p.nsub = 0;
    idx64[0] = INT64_MIN;
    idx64[1] = INT64_MAX;
    for (size_t i = 2; i < 64; ++i)
      idx64[i] = (int64_t)i;
    CHECK(quant_decode(f64back, idx64, &p, Q_F64, 64) == 0);
    for (size_t i = 0; i < 64; ++i)
      CHECK(isfinite(f64back[i]));
  }

  // Enough decades to push exp past its range while the product with the anchor
  // is still representable, and the index past what a double counts exactly.
  {
    char err[256];
    quant_spec sp;
    quant_params p;
    double lo, hi;
    int neg;
    for (size_t i = 0; i < N; ++i)
      f64src[i] = pow(10.0, -47.0 + 308.0 * (double)i / (double)N);
    CHECK(quant_spec_parse("rel:33.33%", &sp, err, sizeof(err)) == 0);
    CHECK(quant_scan(f64src, Q_F64, N, &lo, &hi, &neg) == 0);
    CHECK(quant_spec_resolve(&sp, Q_F64, lo, hi, neg, &p, err, sizeof(err)) ==
          0);
    CHECK(quant_encode(idx64, f64src, &p, Q_F64, N) == 0);
    CHECK(quant_decode(f64back, idx64, &p, Q_F64, N) == 0);
    for (size_t i = 0; i < N; ++i) {
      CHECK(isfinite(f64back[i]));
      CHECK(fabs(f64src[i] - f64back[i]) <= 0.3333 * fabs(f64src[i]));
    }

    // and the bound that needs more levels than a double counts is refused,
    // rather than handed to a search that cannot move by one
    CHECK(quant_spec_parse("shot:a=100,b=1,k=0.5", &sp, err, sizeof(err)) == 0);
    CHECK(quant_spec_resolve(&sp, Q_F64, 1.0, 1.0e31, 0, &p, err,
                             sizeof(err)) != 0);
  }

  // The encoder measures its own round trip rather than trusting the algebra
  // that produced the parameters, so a coarsened grid has to show up as a
  // number above one and an untouched one has to not.
  {
    char err[256];
    quant_spec sp;
    quant_params p;
    double lo, hi, worst;
    int neg;
    for (size_t i = 0; i < N; ++i)
      f32src[i] = (float)pow(10.0, -6.0 + 12.0 * (double)i / (double)N);
    CHECK(quant_spec_parse("rel:1%", &sp, err, sizeof(err)) == 0);
    CHECK(quant_scan(f32src, Q_F32, N, &lo, &hi, &neg) == 0);
    CHECK(quant_spec_resolve(&sp, Q_F32, lo, hi, neg, &p, err, sizeof(err)) ==
          0);
    CHECK(quant_encode(idx32, f32src, &p, Q_F32, N) == 0);
    CHECK(quant_decode(f32back, idx32, &p, Q_F32, N) == 0);
    CHECK(quant_verify(f32src, f32back, &sp, Q_F32, N, &worst) == 0);
    CHECK(worst <= 1.0);

    const quant_params tight = p;
    p.step *= 3.0;
    CHECK(quant_encode(idx32, f32src, &p, Q_F32, N) == 0);
    CHECK(quant_decode(f32back, idx32, &p, Q_F32, N) == 0);
    CHECK(quant_verify(f32src, f32back, &sp, Q_F32, N, &worst) != 0);
    CHECK(worst > 2.5);

    // and the correction the encoder applies brings it back under, from the
    // same call the encoder makes, so this covers the loop and not a copy of it
    CHECK(quant_fit(idx32, f32back, f32src, &sp, &p, Q_F32, N) == 0);
    CHECK(quant_verify(f32src, f32back, &sp, Q_F32, N, &worst) == 0);
    CHECK(p.step <= tight.step);

    // lossless is a byte comparison, since no bound is declared to scale by
    CHECK(quant_spec_parse(NULL, &sp, err, sizeof(err)) == 0);
    CHECK(quant_spec_resolve(&sp, Q_F32, lo, hi, neg, &p, err, sizeof(err)) ==
          0);
    CHECK(quant_encode(idx32, f32src, &p, Q_F32, N) == 0);
    CHECK(quant_decode(f32back, idx32, &p, Q_F32, N) == 0);
    CHECK(quant_verify(f32src, f32back, &sp, Q_F32, N, &worst) == 0);
    f32back[7] += 1.0f;
    CHECK(quant_verify(f32src, f32back, &sp, Q_F32, N, &worst) != 0);
  }

  // An integer input whose values do not span much carries its whole forward
  // map in a table, which the encoder builds once above a size threshold.
  // Either side of that threshold the output has to be the same byte for byte,
  // or the threshold would quietly change what a frame holds. The span comes
  // from the data, so a 32 bit type over a narrow range gets the table too.
  {
    char err[256];
    quant_spec sp;
    quant_params p;
    static uint32_t big[1 << 20];
    static uint32_t viaTable[1 << 20];
    uint32_t direct[N];
    const char *recipes[] = {"shot:a=100,b=1,k=0.5", "rel:1%"};
    for (size_t r = 0; r < 2; ++r) {
      double lo, hi;
      int neg;
      for (size_t i = 0; i < (1u << 20); ++i)
        big[i] = (uint32_t)((i * 7919u) % 60000u + 1u);
      CHECK(quant_spec_parse(recipes[r], &sp, err, sizeof(err)) == 0);
      CHECK(quant_scan(big, Q_U32, 1u << 20, &lo, &hi, &neg) == 0);
      CHECK(quant_spec_resolve(&sp, Q_U32, lo, hi, neg, &p, err, sizeof(err)) ==
            0);
      CHECK(quant_encode(direct, big, &p, Q_U32, N) == 0);
      CHECK(quant_encode(viaTable, big, &p, Q_U32, 1u << 20) == 0);
      CHECK(memcmp(direct, viaTable, N * sizeof(uint32_t)) == 0);
    }
  }

  // The decoder has the mirror of that table, keyed on the index range in the
  // stream, and the same rule applies to it: either side of the size at which
  // it stops building one, the frame has to decode to the same bytes. The
  // stream comes off a frame that may be damaged, so the range it declares is
  // not bounded by anything the encoder would have written.
  {
    quant_params p;
    memset(&p, 0, sizeof(p));
    p.curve = QUANT_CURVE_LOG;
    p.step = 0.01;
    p.offset = 1.0;

    // A u64 index above INT64_MAX, where reading the range back as a signed
    // integer keys the table on a different number than the direct path uses.
    static uint64_t u64idx[N + 1], u64lut[N + 1], u64dir[N + 1];
    for (size_t i = 0; i < N; ++i)
      u64idx[i] = (uint64_t)INT64_MAX + 1u + i % 64u;
    CHECK(quant_decode(u64lut, u64idx, &p, Q_U64, N) == 0);
    u64idx[N] = (uint64_t)INT64_MAX + 1u + 20000u; // span past the table
    CHECK(quant_decode(u64dir, u64idx, &p, Q_U64, N + 1) == 0);
    CHECK(memcmp(u64lut, u64dir, N * sizeof(uint64_t)) == 0);

    p.curve = QUANT_CURVE_SQRT;
    p.step = 0.3;
    p.offset = 12.0;
    static int32_t i32idx[N + 1], i32lut[N + 1], i32dir[N + 1];
    for (size_t i = 0; i < N; ++i)
      i32idx[i] = (int32_t)(i % 64u) - 32;
    CHECK(quant_decode(i32lut, i32idx, &p, Q_I32, N) == 0);
    i32idx[N] = 20000;
    CHECK(quant_decode(i32dir, i32idx, &p, Q_I32, N + 1) == 0);
    CHECK(memcmp(i32lut, i32dir, N * sizeof(int32_t)) == 0);
  }

  if (failures == 0)
    printf("  ok\n");
  return failures != 0;
}