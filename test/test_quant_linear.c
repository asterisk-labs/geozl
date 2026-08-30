#include "quant_linear/decode_quant_linear_kernel.h"
#include "quant_linear/encode_quant_linear_kernel.h"
#include "quant_linear/quant_linear_dtype.h"
#include "quant_linear/quant_linear_half.h"
#include "quant_linear/quant_linear_spec.h"

#include "quant_walk.h"

#include <float.h>
#include <locale.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(c)                                                               \
  do {                                                                         \
    if (!(c)) {                                                                \
      printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #c);                    \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

#define CHECKF(c, ...)                                                         \
  do {                                                                         \
    if (!(c)) {                                                                \
      printf("  FAIL %s:%d  %s  ", __FILE__, __LINE__, #c);                    \
      printf(__VA_ARGS__);                                                     \
      printf("\n");                                                            \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

// One trip the way the encode binding does it, so the step this measures is the
// step a frame would be written with.
static int trip(const char *recipe, int dtype, const void *src, void *dec,
                size_t n, quant_linear_params *pOut) {
  char err[256];
  quant_linear_spec sp;
  if (quant_linear_parse(recipe, &sp, err, sizeof(err)) != 0)
    return -1;
  quant_linear_stats sc;
  quant_linear_scan(src, dtype, n, &sc);
  quant_linear_params p;
  if (quant_linear_resolve(&sp, dtype, &sc, &p, err, sizeof(err)) != 0)
    return -1;
  void *idx = malloc(n * quant_linear_width(dtype));
  int rc = -1;
  if (idx != NULL && quant_linear_encode(idx, src, &p, dtype, n) == 0 &&
      quant_linear_decode(dec, idx, &p, dtype, n) == 0)
    rc = 0;
  free(idx);
  if (pOut != NULL)
    *pOut = p;
  return rc;
}

static void zero_is_a_finite_domain(void) {
  puts("zero is a finite LINEAR domain");
  const uint16_t u16[16] = {0};
  const float f32[16] = {0};
  const float nonfinite[3] = {NAN, INFINITY, -INFINITY};
  uint16_t back[16];
  quant_linear_stats sc;

  CHECK(quant_linear_scan(u16, QL_U16, 16, &sc) == 0);
  CHECK(sc.maxAbs == 0.0 && sc.anyNegative == 0);
  CHECK(quant_linear_scan(f32, QL_F32, 16, &sc) == 0);
  CHECK(sc.maxAbs == 0.0 && sc.anyNegative == 0);
  CHECK(quant_linear_scan(nonfinite, QL_F32, 3, &sc) != 0);
  CHECK(quant_linear_scan(u16, QL_U16, 0, &sc) != 0);

  CHECK(trip("LINEAR:MAX_ERROR=10", QL_U16, u16, back, 16, NULL) == 0);
  CHECK(memcmp(u16, back, sizeof(u16)) == 0);
}

static void an_integer_grid_never_reads_the_tile(void) {
  printf("an integer grid is the same grid whatever the tile holds\n");
  char err[256];
  quant_linear_spec sp;
  quant_linear_params a, b, c;
  CHECK(quant_linear_parse("LINEAR:MAX_ERROR=5", &sp, err, sizeof(err)) == 0);

  static const int dts[] = {QL_U8, QL_U16, QL_I16, QL_I32};
  for (size_t d = 0; d < sizeof(dts) / sizeof(*dts); ++d) {
    CHECK(quant_linear_resolve(&sp, dts[d], &(quant_linear_stats){3.0, 0}, &a,
                               err, sizeof(err)) == 0);
    CHECK(quant_linear_resolve(&sp, dts[d], &(quant_linear_stats){200.0, 0}, &b,
                               err, sizeof(err)) == 0);
    CHECK(quant_linear_resolve(&sp, dts[d], &(quant_linear_stats){100.0, 1}, &c,
                               err, sizeof(err)) == 0);
    CHECK(a.step == b.step);
    CHECK(a.step == c.step);
    CHECK(a.step == 10.0);
    // Integer storage defaults to INDEX.
    CHECK((a.flags & QUANT_LINEAR_FLAG_STORE_VALUES) == 0);
    CHECK((a.flags & QUANT_LINEAR_FLAG_NONNEGATIVE) != 0);
    CHECK((c.flags & QUANT_LINEAR_FLAG_NONNEGATIVE) == 0);

    // Storage changes, the grid does not.
    quant_linear_spec spv;
    quant_linear_params v;
    CHECK(quant_linear_parse("LINEAR:MAX_ERROR=5,STORE=VALUES", &spv, err,
                             sizeof(err)) == 0);
    CHECK(quant_linear_resolve(&spv, dts[d], &(quant_linear_stats){3.0, 0}, &v,
                               err, sizeof(err)) == 0);
    CHECK((v.flags & QUANT_LINEAR_FLAG_STORE_VALUES) != 0);
    CHECK(v.step == a.step);
  }

  // The step never rounds up past the bound.
  CHECK(quant_linear_parse("LINEAR:MAX_ERROR=0.94", &sp, err, sizeof(err)) == 0);
  CHECK(quant_linear_resolve(&sp, QL_U16, &(quant_linear_stats){1000.0, 0}, &a,
                             err, sizeof(err)) == 0);
  CHECK(a.step == 1.0); // floor(1.88), not 2
  CHECK(quant_linear_parse("LINEAR:MAX_ERROR=12.75", &sp, err, sizeof(err)) == 0);
  CHECK(quant_linear_resolve(&sp, QL_I16, &(quant_linear_stats){1000.0, 1}, &a,
                             err, sizeof(err)) == 0);
  CHECK(a.step == 25.0); // floor(25.5), not 26
  // And a bound under half a unit is lossless rather than a step of zero.
  CHECK(quant_linear_parse("LINEAR:MAX_ERROR=0.4", &sp, err, sizeof(err)) == 0);
  CHECK(quant_linear_resolve(&sp, QL_U8, &(quant_linear_stats){200.0, 0}, &a,
                             err, sizeof(err)) == 0);
  CHECK(a.step == 1.0);
}

static void the_float_index_path_reads_the_tile(void) {
  printf("a float index grid follows the tile\n");
  char err[256];
  quant_linear_spec sp;
  quant_linear_params a, b;
  CHECK(quant_linear_parse("LINEAR:MAX_ERROR=0.05", &sp, err, sizeof(err)) == 0);
  // The resolver chooses the dtype-specific default.
  CHECK(sp.store == QUANT_LINEAR_STORE_DEFAULT);
  CHECK(quant_linear_resolve(&sp, QL_F32, &(quant_linear_stats){100.0, 0}, &a,
                             err, sizeof(err)) == 0);
  CHECK(quant_linear_resolve(&sp, QL_F32, &(quant_linear_stats){300.0, 0}, &b,
                             err, sizeof(err)) == 0);
  CHECK(a.step != b.step); // the documented limitation of this path
  CHECK((a.flags & QUANT_LINEAR_FLAG_STORE_VALUES) == 0);

  // A bound below what the output type resolves is refused rather than silently
  // returning the data untouched. At a million a float32 only resolves 0.0625.
  CHECK(quant_linear_parse("LINEAR:MAX_ERROR=1e-5", &sp, err, sizeof(err)) == 0);
  CHECK(quant_linear_resolve(&sp, QL_F32, &(quant_linear_stats){1e5, 0}, &a,
                             err, sizeof(err)) != 0);
}

static void the_float_value_path_does_not(void) {
  printf("a float value grid does not, and is refused rather than falling back\n");
  char err[256];
  quant_linear_spec sp;
  quant_linear_params a, b;
  CHECK(quant_linear_parse("LINEAR:MAX_ERROR=0.5,STORE=VALUES", &sp, err,
                           sizeof(err)) == 0);
  CHECK(sp.store == QUANT_LINEAR_STORE_VALUES);
  CHECK(quant_linear_resolve(&sp, QL_F32, &(quant_linear_stats){100.0, 0}, &a,
                             err, sizeof(err)) == 0);
  CHECK(quant_linear_resolve(&sp, QL_F32, &(quant_linear_stats){30000.0, 0}, &b,
                             err, sizeof(err)) == 0);
  CHECK(a.step == b.step);
  CHECK(a.step == 1.0);
  CHECK((a.flags & QUANT_LINEAR_FLAG_STORE_VALUES) != 0);

  // A bound too tight for a whole step. This is why it cannot be automatic:
  // reflectance in 0 to 1 never gets a whole step, and a fallback would put the
  // dependence on the tile back in.
  quant_linear_spec fine;
  CHECK(quant_linear_parse("LINEAR:MAX_ERROR=0.0005,STORE=VALUES", &fine, err,
                           sizeof(err)) == 0);
  CHECK(quant_linear_resolve(&fine, QL_F32, &(quant_linear_stats){1.0, 0}, &a,
                             err, sizeof(err)) != 0);
  // Past 2^24 a float32 no longer holds every integer, so the cast back would
  // round.
  CHECK(quant_linear_resolve(&sp, QL_F32, &(quant_linear_stats){3e7, 0}, &a,
                             err, sizeof(err)) != 0);
  // On an integer type the reconstruction is what the stream always carries, so
  // asking for it changes nothing.
  CHECK(quant_linear_resolve(&sp, QL_U16, &(quant_linear_stats){60000.0, 0}, &a,
                             err, sizeof(err)) == 0);
  CHECK(a.step == 1.0);
}

static void an_unrepresentable_step_is_refused(void) {
  puts("a grid step has to fit in its header");
  char err[256];
  quant_linear_spec sp;
  quant_linear_params p;
  const quant_linear_stats sc = {100.0, 0};

  CHECK(quant_linear_parse("LINEAR:MAX_ERROR=1e308", &sp, err, sizeof(err)) ==
        0);
  CHECK(quant_linear_resolve(&sp, QL_U16, &sc, &p, err, sizeof(err)) != 0);
  CHECK(quant_linear_resolve(&sp, QL_F32, &sc, &p, err, sizeof(err)) != 0);

  CHECK(quant_linear_parse("LINEAR:MAX_ERROR=1e308,STORE=VALUES", &sp, err,
                           sizeof(err)) == 0);
  CHECK(quant_linear_resolve(&sp, QL_F32, &sc, &p, err, sizeof(err)) != 0);
}

// What STORE=VALUES is for. A DEM read through a masked reader arrives as float32
// with whole metres in it.
static void a_float_array_of_integers_round_trips_exactly(void) {
  printf("a float array holding whole numbers comes back exactly\n");
  enum { N = 2048 };
  static float src[N], dec[N];
  for (size_t i = 0; i < N; ++i)
    src[i] = (float)((int)(i % 4000) - 420);

  quant_linear_params p;
  CHECK(trip("LINEAR:MAX_ERROR=0.5,STORE=VALUES", QL_F32, src, dec, N, &p) == 0);
  CHECK(p.step == 1.0);
  size_t exact = 0;
  for (size_t i = 0; i < N; ++i)
    if (src[i] == dec[i])
      ++exact;
  CHECK(exact == N); // already on the grid, so nothing is lost

  // Temperature, where the data is not integral and the bound is being spent.
  static float t[N], dt[N];
  for (size_t i = 0; i < N; ++i)
    t[i] = 253.15f + (float)(i % 40);
  CHECK(trip("LINEAR:MAX_ERROR=0.5,STORE=VALUES", QL_F32, t, dt, N, &p) == 0);
  for (size_t i = 0; i < N; ++i) {
    CHECK(fabs((double)t[i] - (double)dt[i]) <= 0.5);
    CHECK(dt[i] == nearbyintf(dt[i]));
  }
}

static void non_negative_stays_non_negative(void) {
  printf("a tile with nothing negative decodes to nothing negative\n");
  enum { N = 2048 };
  static float src[N], dec[N];
  for (size_t i = 0; i < N; ++i)
    src[i] = i < 256 ? 0.0f : (float)(i % 500) / 100.0f;

  quant_linear_params p;
  CHECK(trip("LINEAR:MAX_ERROR=0.05", QL_F32, src, dec, N, &p) == 0);
  CHECK((p.flags & QUANT_LINEAR_FLAG_NONNEGATIVE) != 0);
  size_t below = 0, zeros = 0, exact = 0;
  for (size_t i = 0; i < N; ++i) {
    if (dec[i] < 0.0f)
      ++below;
    if (src[i] == 0.0f) {
      ++zeros;
      if (dec[i] == 0.0f)
        ++exact;
    }
  }
  CHECK(below == 0);
  CHECK(zeros >= 256);
  CHECK(exact == zeros);

  // A tile that does hold negatives keeps them, or the floor would be a guess
  // about the data rather than a measurement of it.
  static int16_t s2[N], d2[N];
  for (size_t i = 0; i < N; ++i)
    s2[i] = (int16_t)((int)(i % 800) - 420);
  CHECK(trip("LINEAR:MAX_ERROR=5", QL_I16, s2, d2, N, &p) == 0);
  CHECK((p.flags & QUANT_LINEAR_FLAG_NONNEGATIVE) == 0);
  size_t negatives = 0;
  for (size_t i = 0; i < N; ++i)
    if (d2[i] < 0)
      ++negatives;
  CHECK(negatives > 0);
}

static void the_parser_is_strict(void) {
  printf("the parser refuses what it cannot mean\n");
  char err[256];
  quant_linear_spec sp;
  CHECK(quant_linear_parse("LINEAR:MAX_ERROR=5", &sp, err, sizeof(err)) == 0);
  CHECK(quant_linear_parse("LINEAR:MAX_ERROR=5,STORE=INDEX", &sp, err,
                           sizeof(err)) == 0);
  CHECK(quant_linear_parse("LINEAR:STORE=VALUES,MAX_ERROR=5", &sp, err,
                           sizeof(err)) == 0);
  CHECK(quant_linear_parse(NULL, &sp, err, sizeof(err)) != 0);
  CHECK(quant_linear_parse("", &sp, err, sizeof(err)) != 0);
  CHECK(quant_linear_parse("LINEAR:", &sp, err, sizeof(err)) != 0);
  CHECK(quant_linear_parse("linear:MAX_ERROR=5", &sp, err, sizeof(err)) != 0);
  CHECK(quant_linear_parse("LINEAR:STORE=VALUES", &sp, err, sizeof(err)) != 0);
  CHECK(quant_linear_parse("LINEAR:MAX_ERROR=0", &sp, err, sizeof(err)) != 0);
  CHECK(quant_linear_parse("LINEAR:MAX_ERROR=-1", &sp, err, sizeof(err)) != 0);
  CHECK(quant_linear_parse("LINEAR:MAX_ERROR=nan", &sp, err, sizeof(err)) != 0);
  CHECK(quant_linear_parse("LINEAR:MAX_ERROR=1.0.0", &sp, err, sizeof(err)) != 0);
  CHECK(quant_linear_parse("LINEAR:MAX_ERROR=", &sp, err, sizeof(err)) != 0);
  CHECK(quant_linear_parse("LINEAR:MAX_ERROR=5,MAX_ERROR=6", &sp, err,
                           sizeof(err)) != 0);
  CHECK(quant_linear_parse("LINEAR:MAX_ERROR=5,STORE=NOPE", &sp, err,
                           sizeof(err)) != 0);
  CHECK(quant_linear_parse("LINEAR:MAX_ERROR=5,ZZZ=1", &sp, err, sizeof(err)) !=
        0);
  CHECK(quant_linear_parse("LINEAR:MAX_ERROR=5,", &sp, err, sizeof(err)) != 0);
  CHECK(quant_linear_parse("RELATIVE:MAX_ERROR=5", &sp, err, sizeof(err)) != 0);
}

static void a_forged_header_is_refused(void) {
  printf("the kernels refuse a parameter block the resolver cannot produce\n");
  enum { N = 64 };
  static float f[N], df[N];
  static uint16_t u[N], du[N];
  quant_linear_params p;
  memset(&p, 0, sizeof(p));

  p.step = 0.0;
  CHECK(quant_linear_decode(df, f, &p, QL_F32, N) != 0);
  p.step = -1.0;
  CHECK(quant_linear_decode(df, f, &p, QL_F32, N) != 0);

  // A stored reconstruction on a float type is legal, but only with a whole step.
  p.step = 1.0;
  p.flags = QUANT_LINEAR_FLAG_STORE_VALUES;
  CHECK(quant_linear_decode(df, f, &p, QL_F32, N) == 0);
  p.step = 0.5;
  CHECK(quant_linear_decode(df, f, &p, QL_F32, N) != 0);

  // Both integer storage modes require a whole step.
  p.step = 1.0;
  p.flags = 0;
  CHECK(quant_linear_decode(du, u, &p, QL_U16, N) == 0);
  p.step = 10.0;
  CHECK(quant_linear_decode(du, u, &p, QL_U16, N) == 0);
  p.step = 0.5;
  CHECK(quant_linear_decode(du, u, &p, QL_U16, N) != 0);
  CHECK(quant_linear_encode(du, u, &p, QL_U16, N) != 0);
  p.flags = QUANT_LINEAR_FLAG_STORE_VALUES;
  CHECK(quant_linear_decode(du, u, &p, QL_U16, N) != 0);

  p.flags = QUANT_LINEAR_FLAG_STORE_VALUES;
  CHECK(quant_linear_decode(du, u, &p, 99, N) != 0);
  CHECK(quant_linear_encode(du, u, &p, -1, N) != 0);
}

static void the_edges(void) {
  printf("constant tiles, all zeros, one element, type limits\n");
  enum { N = 512 };
  static uint16_t u[N], du[N];
  static float f[N], df[N];
  quant_linear_params p;

  for (size_t i = 0; i < N; ++i)
    u[i] = 1234;
  CHECK(trip("LINEAR:MAX_ERROR=5", QL_U16, u, du, N, &p) == 0);
  for (size_t i = 0; i < N; ++i)
    CHECK(abs((int)du[i] - 1234) <= 5);

  for (size_t i = 0; i < N; ++i)
    u[i] = 65535;
  CHECK(trip("LINEAR:MAX_ERROR=5", QL_U16, u, du, N, &p) == 0);
  for (size_t i = 0; i < N; ++i)
    CHECK(65535 - du[i] <= 5);

  memset(f, 0, sizeof(f));
  CHECK(trip("LINEAR:MAX_ERROR=0.5", QL_F32, f, df, N, &p) == 0);
  for (size_t i = 0; i < N; ++i)
    CHECK(df[i] == 0.0f);

  float one = 7.5f, done = 0.0f;
  CHECK(trip("LINEAR:MAX_ERROR=0.5", QL_F32, &one, &done, 1, &p) == 0);
  CHECK(fabs((double)done - 7.5) <= 0.5);

  // The nodata codec in front of a lossy graph owns these, so all this has to do
  // is not fall over.
  for (size_t i = 0; i < N; ++i)
    f[i] = NAN;
  CHECK(trip("LINEAR:MAX_ERROR=0.5", QL_F32, f, df, N, &p) == 0);
}

static void the_parser_takes_only_decimals(void) {
  puts("the parser takes only what the grammar writes");
  quant_linear_spec sp;
  char err[256];

#define GOOD(s) CHECK(quant_linear_parse((s), &sp, err, sizeof(err)) == 0)
#define BAD(s) CHECK(quant_linear_parse((s), &sp, err, sizeof(err)) != 0)

  GOOD("LINEAR:MAX_ERROR=0.5");
  GOOD("LINEAR:MAX_ERROR=5");
  GOOD("LINEAR:MAX_ERROR=1e-3");
  GOOD("LINEAR:MAX_ERROR=1E+3");
  GOOD("LINEAR:MAX_ERROR=.5");
  GOOD("LINEAR:MAX_ERROR=2,STORE=VALUES");

  BAD("LINEAR:MAX_ERROR=inf");
  BAD("LINEAR:MAX_ERROR=nan");
  BAD("LINEAR:MAX_ERROR=0.5 ");
  BAD("LINEAR:MAX_ERROR=1e");
  BAD("LINEAR:MAX_ERROR=1.0.0");
  BAD("LINEAR:MAX_ERROR=1_000");
  BAD("LINEAR:MAX_ERROR=1e999");
  BAD("LINEAR:MAX_ERROR=1e-320"); // underflows to subnormal

#undef GOOD
#undef BAD
}

// Skipped where no comma locale is installed, which is most containers.
static void a_comma_locale_reads_the_same_recipe(void) {
  puts("a comma locale reads the same recipe");
  static const char *names[] = {"de_DE.UTF-8", "fr_FR.UTF-8", "es_PE.UTF-8",
                                "de_DE", "fr_FR"};
  const char *got = NULL;
  for (size_t i = 0; i < sizeof(names) / sizeof(names[0]) && got == NULL; ++i)
    got = setlocale(LC_NUMERIC, names[i]);

  if (got == NULL) {
    puts("  no comma locale installed, skipped");
    return;
  }
  printf("  under %s\n", got);

  quant_linear_spec sp;
  char err[256];
  CHECK(quant_linear_parse("LINEAR:MAX_ERROR=0.5", &sp, err, sizeof(err)) == 0);
  CHECK(sp.max_error == 0.5);
  CHECK(quant_linear_parse("LINEAR:MAX_ERROR=0,5", &sp, err, sizeof(err)) != 0);

  setlocale(LC_NUMERIC, "C");
}

static void verify_sees_the_bound_hold(void) {
  puts("verify measures the round trip against the bound");
  enum { N = 4096 };
  static const struct {
    const char *name;
    const char *recipe;
    int dtype;
  } cases[] = {
      {"u16 DN", "LINEAR:MAX_ERROR=5", QL_U16},
      {"i16 DEM", "LINEAR:MAX_ERROR=8", QL_I16},
      {"f32 index", "LINEAR:MAX_ERROR=0.05", QL_F32},
      {"f32 values", "LINEAR:MAX_ERROR=2,STORE=VALUES", QL_F32},
      {"f64 index", "LINEAR:MAX_ERROR=0.001", QL_F64},
      {"f16 index", "LINEAR:MAX_ERROR=0.05", QL_F16},
  };

  for (size_t k = 0; k < sizeof(cases) / sizeof(cases[0]); ++k) {
    const size_t w = quant_linear_width(cases[k].dtype);
    void *src = calloc(N, w), *mid = calloc(N, w), *dec = calloc(N, w);
    CHECK(src != NULL && mid != NULL && dec != NULL);
    if (src == NULL || mid == NULL || dec == NULL) {
      free(src);
      free(mid);
      free(dec);
      continue;
    }

    for (size_t i = 0; i < N; ++i) {
      const double v = 100.0 + 300.0 * (double)(i % 211) / 211.0;
      switch (cases[k].dtype) {
      case QL_U16:
        ((uint16_t *)src)[i] = (uint16_t)(v * 100.0);
        break;
      case QL_I16:
        ((int16_t *)src)[i] = (int16_t)(v - 200.0);
        break;
      case QL_F32:
        ((float *)src)[i] = (float)v;
        break;
      case QL_F64:
        ((double *)src)[i] = v;
        break;
      default:
        ((uint16_t *)src)[i] = 0x5000u + (uint16_t)(i % 64);
        break;
      }
    }

    char err[256];
    quant_linear_spec sp;
    quant_linear_params p;
    double worst = -1.0;
    quant_linear_stats sc;
    CHECK(quant_linear_parse(cases[k].recipe, &sp, err, sizeof(err)) == 0);
    quant_linear_scan(src, cases[k].dtype, N, &sc);
    CHECK(quant_linear_resolve(&sp, cases[k].dtype, &sc, &p, err,
                               sizeof(err)) == 0);
    CHECK(quant_linear_encode(mid, src, &p, cases[k].dtype, N) == 0);
    CHECK(quant_linear_decode(dec, mid, &p, cases[k].dtype, N) == 0);
    CHECK(quant_linear_verify(src, dec, &sp, cases[k].dtype, N, &worst) == 0);
    if (!(worst <= 1.0))
      printf("  FAIL %s: worst %.6f of the budget\n", cases[k].name, worst);
    CHECK(worst <= 1.0);
    printf("  %-12s used %.6f of the budget\n", cases[k].name, worst);

    free(src);
    free(mid);
    free(dec);
  }
}

// verify has to be able to say no, or it only ever confirms itself.
static void verify_sees_the_bound_break(void) {
  puts("verify reports a reconstruction that is out of bounds");
  enum { N = 4 };
  const float src[N] = {10.0f, 20.0f, 30.0f, 40.0f};
  const float dec[N] = {10.0f, 20.0f, 34.0f, 40.0f};
  quant_linear_spec sp;
  memset(&sp, 0, sizeof(sp));
  sp.max_error = 2.0;
  double worst = -1.0;
  CHECK(quant_linear_verify(src, dec, &sp, QL_F32, N, &worst) == 0);
  CHECK(worst == 2.0); // off by four against a bound of two

  const float withNan[N] = {10.0f, NAN, 30.0f, 40.0f};
  const float back[N] = {10.0f, 0.0f, 30.0f, 40.0f};
  worst = -1.0;
  CHECK(quant_linear_verify(withNan, back, &sp, QL_F32, N, &worst) == 0);
  CHECK(worst == 0.0); // the non-finite sample belongs to nodata
}

// The branchless half conversions against the reference, over their whole
// domain. The first draft of ql_f32_to_half was wrong on two of the 2^32, both
// where the subnormal rounding carries into the smallest normal, and no sampled
// test would have found them.
//
static void the_half_conversions_match_the_reference(void) {
  const uint64_t step = geozl_walk_step();
  printf("the half conversions match the reference, %s\n",
         step == 1 ? "every value" : "on a stride");

  uint64_t bad = 0;
  for (int32_t v = -32768; v <= 32767; ++v)
    if (quant_linear_float_to_half((float)v) != ql_i16_to_half(v) && bad++ < 8)
      printf("  ql_i16_to_half at %d\n", v);
  CHECK(bad == 0);

  bad = 0;
  uint64_t seen = 0;
  for (uint64_t u = 0; u <= 0xFFFFFFFFull; u += step) {
    const uint32_t bits = (uint32_t)u;
    float f;
    memcpy(&f, &bits, sizeof(f));
    ++seen;
    if (quant_linear_float_to_half(f) != ql_f32_to_half(f) && bad++ < 8)
      printf("  ql_f32_to_half at 0x%08x\n", bits);
  }
  printf("  %llu float32 checked\n", (unsigned long long)seen);
  CHECK(bad == 0);
}

// Every value of a type through the grid, one at a time. The small domains are
// walked whole; f32 goes on a stride unless GEOZL_EXHAUSTIVE is set.
static void every_value_holds_the_bound(void) {
  static const char *const recipes[] = {"LINEAR:MAX_ERROR=5",
                                        "LINEAR:MAX_ERROR=0.94",
                                        "LINEAR:MAX_ERROR=0.25"};
  static const double bounds[] = {5.0, 0.94, 0.25};
  static uint8_t in[GEOZL_WALK_BLK * 4], st[GEOZL_WALK_BLK * 4];
  static uint8_t out[GEOZL_WALK_BLK * 4];
  const uint64_t step = geozl_walk_step();
  printf("every value of a type holds the bound, f32 %s\n",
         step == 1 ? "whole" : "on a stride");

  for (size_t t = 0; t < GEOZL_WALK_NTYPES; ++t) {
    const int dt = geozl_walk_types[t].dtype;
    for (size_t r = 0; r < sizeof(bounds) / sizeof(bounds[0]); ++r) {
      char err[256];
      quant_linear_spec sp;
      quant_linear_params p;
      CHECK(quant_linear_parse(recipes[r], &sp, err, sizeof err) == 0);
      // The grid is cut against the widest the type reaches and the sign it
      // carries, since the walk covers the whole domain.
      // The grid is cut against a range a raster holds, not against the whole
      // float domain, which no uniform grid spans. Values past it are skipped
      // and the count below says how much of the domain that left.
      static const double tops[] = {255.0, 65535.0, 32767.0, 1e4, 1e4};
      const int neg = dt == GEOZL_DT_I16 || dt == GEOZL_DT_F16 ||
                      dt == GEOZL_DT_F32;
      const quant_linear_stats sc = {tops[t], neg};
      if (quant_linear_resolve(&sp, dt, &sc, &p, err, sizeof err) != 0) {
        // A refusal is an answer. No uniform grid spans the whole float domain,
        // so this is where the codec says that out loud.
        printf("  %-4s %-22s refused, %s\n", geozl_walk_types[t].name,
               recipes[r], err);
        continue;
      }

      double worst = 0.0;
      uint64_t counted = 0, over = 0, sum = 0, at = 0;
      const uint64_t domain = geozl_walk_types[t].domain;
      const uint64_t span = dt == GEOZL_DT_F32 ? step : 1;
      for (uint64_t base = 0; base < domain; base += GEOZL_WALK_BLK * span) {
        const size_t n = geozl_walk_fill(in, dt, base, domain, span);
        CHECK(quant_linear_encode(st, in, &p, dt, n) == 0);
        CHECK(quant_linear_decode(out, st, &p, dt, n) == 0);
        for (size_t i = 0; i < n; ++i) {
          const uint64_t pos = base + (uint64_t)i * span;
          GEOZL_FOLD(sum, geozl_walk_bits(out, dt, i), pos);
          const double x = geozl_walk_get(in, dt, i);
          const double y = geozl_walk_get(out, dt, i);
          if (!isfinite(x))
            continue; // the nodata codec owns these
          if (fabs(x) > tops[t])
            continue; // outside the range the grid was cut for
          ++counted;
          const double e = fabs(y - x);
          if (e > worst) {
            worst = e;
            at = pos;
          }
          if (e > bounds[r])
            ++over;
        }
      }
      printf("  %-4s %-22s %10llu values, worst %.4f of the bound, sum %016llx\n",
             geozl_walk_types[t].name, recipes[r],
             (unsigned long long)counted, worst / bounds[r],
             (unsigned long long)sum);
      if (over != 0) {
        printf("    FAIL %llu over the bound, worst at 0x%llx\n",
               (unsigned long long)over, (unsigned long long)at);
        ++failures;
      }
    }
  }
}

// Exhaustive walks stop at 16 bits; cover the wide limits explicitly.
static void the_wide_integers_hold_at_the_corners(void) {
  printf("the wide integer types hold at their corners\n");
  static const double steps[] = {1.0, 2.0, 7.0, 1000.0, 4294967296.0};

  for (size_t k = 0; k < sizeof(steps) / sizeof(*steps); ++k) {
    quant_linear_params p = {0, steps[k]};

    {
      const uint64_t in[6] = {0u, 1u, 1000u, UINT64_MAX - 1u, UINT64_MAX,
                              UINT32_MAX};
      uint64_t st[6], back[6];
      CHECK(quant_linear_encode(st, in, &p, QL_U64, 6) == 0);
      CHECK(quant_linear_decode(back, st, &p, QL_U64, 6) == 0);
      for (size_t i = 0; i < 6; ++i) {
        CHECKF(st[i] <= in[i], "u64 index %llu above its sample %llu",
               (unsigned long long)st[i], (unsigned long long)in[i]);
        // Avoid unsigned underflow in the difference.
        const uint64_t diff =
            back[i] > in[i] ? back[i] - in[i] : in[i] - back[i];
        CHECKF(diff <= (uint64_t)steps[k] || back[i] == UINT64_MAX,
               "u64 %llu rebuilt as %llu at step %g",
               (unsigned long long)in[i], (unsigned long long)back[i],
               steps[k]);
      }
    }
    {
      const int64_t in[7] = {0, 1, -1, 1000, INT64_MIN, INT64_MAX, INT32_MIN};
      int64_t st[7], back[7];
      CHECK(quant_linear_encode(st, in, &p, QL_I64, 7) == 0);
      CHECK(quant_linear_decode(back, st, &p, QL_I64, 7) == 0);
      for (size_t i = 0; i < 7; ++i) {
        CHECKF((in[i] < 0) == (back[i] < 0) || back[i] == 0,
               "i64 %lld came back as %lld with the sign moved",
               (long long)in[i], (long long)back[i]);
        CHECKF(back[i] >= INT64_MIN && back[i] <= INT64_MAX,
               "i64 %lld left its type", (long long)back[i]);
      }
    }
    {
      const uint32_t in[4] = {0u, 1u, UINT32_MAX - 1u, UINT32_MAX};
      uint32_t st[4], back[4];
      CHECK(quant_linear_encode(st, in, &p, QL_U32, 4) == 0);
      CHECK(quant_linear_decode(back, st, &p, QL_U32, 4) == 0);
      for (size_t i = 0; i < 4; ++i)
        CHECKF(st[i] <= in[i], "u32 index %u above its sample %u", st[i], in[i]);
    }
    {
      const int32_t in[5] = {0, -1, 1, INT32_MIN, INT32_MAX};
      int32_t st[5], back[5];
      CHECK(quant_linear_encode(st, in, &p, QL_I32, 5) == 0);
      CHECK(quant_linear_decode(back, st, &p, QL_I32, 5) == 0);
      for (size_t i = 0; i < 5; ++i)
        CHECKF((in[i] < 0) == (back[i] < 0) || back[i] == 0,
               "i32 %d came back as %d with the sign moved", in[i], back[i]);
    }
  }

  // Saturation must precede a product that would overflow 128 bits.
  {
    quant_linear_params p = {0, 18446744073709549568.0};
    const uint64_t in[2] = {UINT64_MAX, UINT64_MAX / 2u};
    uint64_t back[2];
    CHECK(quant_linear_decode(back, in, &p, QL_U64, 2) == 0);
    CHECK(back[0] == UINT64_MAX && back[1] == UINT64_MAX);
  }
  {
    quant_linear_params p = {0, 9223372036854775808.0};
    const int64_t in[2] = {INT64_MAX, INT64_MIN};
    int64_t back[2];
    CHECK(quant_linear_decode(back, in, &p, QL_I64, 2) == 0);
    CHECK(back[0] == INT64_MAX && back[1] == INT64_MIN);
  }
}

int main(void) {
  zero_is_a_finite_domain();
  an_integer_grid_never_reads_the_tile();
  the_float_index_path_reads_the_tile();
  the_float_value_path_does_not();
  an_unrepresentable_step_is_refused();
  a_float_array_of_integers_round_trips_exactly();
  non_negative_stays_non_negative();
  the_parser_is_strict();
  the_parser_takes_only_decimals();
  a_comma_locale_reads_the_same_recipe();
  a_forged_header_is_refused();
  verify_sees_the_bound_hold();
  verify_sees_the_bound_break();
  the_edges();
  the_half_conversions_match_the_reference();
  the_wide_integers_hold_at_the_corners();
  every_value_holds_the_bound();
  if (failures != 0) {
    printf("%d failed\n", failures);
    return 1;
  }
  printf("all passed\n");
  return 0;
}
