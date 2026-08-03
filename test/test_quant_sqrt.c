#include "quant_sqrt/decode_quant_sqrt_kernel.h"
#include "quant_sqrt/encode_quant_sqrt_kernel.h"
#include "quant_sqrt/quant_sqrt_dtype.h"
#include "quant_sqrt/quant_sqrt_half.h"
#include "quant_sqrt/quant_sqrt_spec.h"

#include <float.h>
#include <locale.h>
#include "quant_walk.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

// The budget sweep walks into the refusal region on purpose.
static int report_refusals = 1;

#define CHECK(c)                                                               \
  do {                                                                         \
    if (!(c)) {                                                                \
      printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #c);                    \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

#define CHECKF(c, fmt, ...)                                                    \
  do {                                                                         \
    if (!(c)) {                                                                \
      printf("  FAIL %s:%d  %s  " fmt "\n", __FILE__, __LINE__, #c,            \
             __VA_ARGS__);                                                     \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

// xorshift64star, so a failure is reproducible from the seed alone.
static uint64_t rng_state = 0x9e3779b97f4a7c15ull;

static double uniform(void) {
  rng_state ^= rng_state >> 12;
  rng_state ^= rng_state << 25;
  rng_state ^= rng_state >> 27;
  const uint64_t v = rng_state * 2685821657736338717ull;
  return (double)(v >> 11) * (1.0 / 9007199254740992.0);
}

// Box Muller, throwing the second away.
static double normal(void) {
  const double u1 = uniform() + 1e-18, u2 = uniform();
  return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

// A smooth radiance field sampled with noise of variance a + b*x, the model the
// codec cuts its grid against. Not clipped to the type, the caller does that.
static void photon_field(double *out, size_t w, size_t h, double top, double a,
                         double b) {
  for (size_t y = 0; y < h; ++y) {
    for (size_t x = 0; x < w; ++x) {
      const double fx = (double)x / (double)w, fy = (double)y / (double)h;
      // two lobes and a ramp, so the histogram covers the range
      double s = 0.25 + 0.45 * fx;
      s += 0.30 * exp(-40.0 * ((fx - 0.30) * (fx - 0.30) +
                               (fy - 0.65) * (fy - 0.65)));
      s -= 0.20 * exp(-25.0 * ((fx - 0.75) * (fx - 0.75) +
                               (fy - 0.25) * (fy - 0.25)));
      double v = s * top;
      if (v < 0.0)
        v = 0.0;
      const double sigma = sqrt(a + b * v);
      v += sigma * normal();
      if (v < 0.0)
        v = 0.0;
      out[y * w + x] = v;
    }
  }
}

static void store_as(void *dst, int dtype, const double *src, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    const double v = src[i];
    switch ((qsq_dtype)dtype) {
    case QSQ_U8:
      ((uint8_t *)dst)[i] = (uint8_t)(v + 0.5);
      break;
    case QSQ_U16:
      ((uint16_t *)dst)[i] = (uint16_t)(v + 0.5);
      break;
    case QSQ_U32:
      ((uint32_t *)dst)[i] = (uint32_t)(v + 0.5);
      break;
    case QSQ_U64:
      ((uint64_t *)dst)[i] = (uint64_t)(v + 0.5);
      break;
    case QSQ_I8:
      ((int8_t *)dst)[i] = (int8_t)(v + 0.5);
      break;
    case QSQ_I16:
      ((int16_t *)dst)[i] = (int16_t)(v + 0.5);
      break;
    case QSQ_I32:
      ((int32_t *)dst)[i] = (int32_t)(v + 0.5);
      break;
    case QSQ_I64:
      ((int64_t *)dst)[i] = (int64_t)(v + 0.5);
      break;
    case QSQ_F16:
      ((uint16_t *)dst)[i] = quant_sqrt_float_to_half((float)v);
      break;
    case QSQ_F32:
      ((float *)dst)[i] = (float)v;
      break;
    default:
      ((double *)dst)[i] = v;
      break;
    }
  }
}

// Read from the recipe, so this agrees with nothing under test.
static double declared(double k, double a, double b, double x) {
  const double v = a + b * x;
  return v > 0.0 ? k * sqrt(v) : 0.0;
}

static double sample_at(const void *p, int dtype, size_t i) {
  switch ((qsq_dtype)dtype) {
  case QSQ_U8:
    return (double)((const uint8_t *)p)[i];
  case QSQ_U16:
    return (double)((const uint16_t *)p)[i];
  case QSQ_U32:
    return (double)((const uint32_t *)p)[i];
  case QSQ_U64:
    return (double)((const uint64_t *)p)[i];
  case QSQ_I8:
    return (double)((const int8_t *)p)[i];
  case QSQ_I16:
    return (double)((const int16_t *)p)[i];
  case QSQ_I32:
    return (double)((const int32_t *)p)[i];
  case QSQ_I64:
    return (double)((const int64_t *)p)[i];
  case QSQ_F16:
    return (double)quant_sqrt_half_to_float(((const uint16_t *)p)[i]);
  case QSQ_F32:
    return (double)((const float *)p)[i];
  default:
    return ((const double *)p)[i];
  }
}

static const char *dtype_name(int d) {
  static const char *n[] = {"u8",  "u16", "u32", "u64", "i8", "i16",
                            "i32", "i64", "f16", "f32", "f64"};
  return QSQ_DTYPE_OK(d) ? n[d] : "?";
}

// Worst error over the declared bound, or -1 when a stage refused.
static double round_trip(int dtype, const char *recipe, const double *field,
                         size_t n, double *codecWorst) {
  char err[256] = {0};
  quant_sqrt_spec sp;
  if (quant_sqrt_parse(recipe, &sp, err, sizeof(err)) != 0) {
    printf("  parse refused %s, %s\n", recipe, err);
    return -1.0;
  }

  const size_t width = quant_sqrt_width(dtype);
  void *src = malloc(n * width), *stream = malloc(n * width),
       *back = malloc(n * width);
  double ratio = -1.0;
  if (src == NULL || stream == NULL || back == NULL)
    goto out;
  store_as(src, dtype, field, n);

  quant_sqrt_stats sc;
  if (quant_sqrt_scan(src, dtype, n, &sc) != 0)
    goto out;

  quant_sqrt_params p;
  if (quant_sqrt_resolve(&sp, dtype, &sc, NULL, &p, err, sizeof(err)) != 0) {
    if (report_refusals)
      printf("  resolve refused %s on %s, %s\n", recipe, dtype_name(dtype), err);
    goto out;
  }
  if (quant_sqrt_encode(stream, src, &p, dtype, n) != 0) {
    printf("  encode refused %s on %s\n", recipe, dtype_name(dtype));
    goto out;
  }
  if (quant_sqrt_decode(back, stream, &p, dtype, n) != 0) {
    printf("  decode refused %s on %s\n", recipe, dtype_name(dtype));
    goto out;
  }

  ratio = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double x = sample_at(src, dtype, i), y = sample_at(back, dtype, i);
    if (x == y)
      continue;
    const double bound = declared(sp.k, sp.a, sp.b, x);
    if (!(bound > 0.0))
      continue;
    const double r = fabs(x - y) / bound;
    if (r > ratio)
      ratio = r;
  }

  if (codecWorst != NULL) {
    *codecWorst = -1.0;
    quant_sqrt_verify(src, back, &sp, NULL, dtype, n, codecWorst);
  }

out:
  free(src);
  free(stream);
  free(back);
  return ratio;
}

static void the_parser_takes_only_valid_recipes(void) {
  puts("the parser takes only valid recipes");

  static const struct {
    const char *recipe;
    int ok;
  } cases[] = {
      {"SQRT:MAX_ERROR=0.5N", 1},
      {"SQRT:MAX_ERROR=0.5N,A=100,B=1", 1},
      {"SQRT:MAX_ERROR=0.5N,B=1,A=100", 1},
      {"SQRT:MAX_ERROR=0.5N,STORE=VALUES", 1},
      {"SQRT:MAX_ERROR=0.5N,A=0,B=1,STORE=INDEX", 1},
      {"SQRT:MAX_ERROR=1N", 1},
      // without the N, half a sigma reads as half a count
      {"SQRT:MAX_ERROR=0.5", 0},
      {"SQRT:MAX_ERROR=0.5n", 0},
      {"SQRT:MAX_ERROR=N", 0},
      {"SQRT:MAX_ERROR=0N", 0},
      {"SQRT:MAX_ERROR=-1N", 0},
      {"SQRT:MAX_ERROR=1.0.0N", 0},
      {"SQRT:MAX_ERROR=0.5N,MAX_ERROR=1N", 0},
      {"SQRT:MAX_ERROR=0.5N,A=100", 0},
      {"SQRT:MAX_ERROR=0.5N,B=1", 0},
      {"SQRT:MAX_ERROR=0.5N,A=-1,B=1", 0},
      {"SQRT:MAX_ERROR=0.5N,A=1,B=0", 0},
      {"SQRT:MAX_ERROR=0.5N,A=1,B=-1", 0},
      {"SQRT:MAX_ERROR=0.5N,STORE=BOTH", 0},
      {"SQRT:MAX_ERROR=0.5N,", 0},
      {"SQRT:MAX_ERROR=0.5N,,A=1", 0},
      {"SQRT:MAX_ERROR=0.5N,SIGMA=1", 0},
      {"LOG:MAX_ERROR=0.5N", 0},
      {"SQRT:", 0},
      {"", 0},
      {NULL, 0},
  };

  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    quant_sqrt_spec sp;
    char err[256] = {0};
    const int r = quant_sqrt_parse(cases[i].recipe, &sp, err, sizeof(err));
    CHECKF(r == (cases[i].ok ? 0 : 1), "%s",
           cases[i].recipe == NULL ? "(null)" : cases[i].recipe);
  }
}

static void a_comma_locale_reads_the_same_recipe(void) {
  puts("a comma locale reads the same recipe");

  quant_sqrt_spec plain;
  CHECK(quant_sqrt_parse("SQRT:MAX_ERROR=0.5N,A=100.25,B=1.5", &plain, NULL,
                         0) == 0);

  const char *got = setlocale(LC_NUMERIC, "de_DE.UTF-8");
  if (got == NULL)
    got = setlocale(LC_NUMERIC, "fr_FR.UTF-8");
  if (got == NULL) {
    puts("  no comma locale on this box, skipped");
    return;
  }

  quant_sqrt_spec under;
  CHECK(quant_sqrt_parse("SQRT:MAX_ERROR=0.5N,A=100.25,B=1.5", &under, NULL,
                         0) == 0);
  CHECK(under.k == plain.k);
  CHECK(under.a == plain.a);
  CHECK(under.b == plain.b);
  setlocale(LC_NUMERIC, "C");
}

// fabs folded the curve at zero and reported a bound the grid never held to.
static void the_curve_is_read_at_signed_x(void) {
  puts("the curve is read at signed x");

  quant_sqrt_spec sp;
  CHECK(quant_sqrt_parse("SQRT:MAX_ERROR=0.5N,A=100,B=1", &sp, NULL, 0) == 0);

  CHECK(fabs(quant_sqrt_bound(&sp, NULL, 0.0) - 5.0) < 1e-12);
  CHECK(fabs(quant_sqrt_bound(&sp, NULL, 300.0) - 10.0) < 1e-12);
  // 0.5*sqrt(100-90) is not 0.5*sqrt(100+90)
  CHECK(fabs(quant_sqrt_bound(&sp, NULL, -90.0) - 0.5 * sqrt(10.0)) < 1e-12);
  CHECK(quant_sqrt_bound(&sp, NULL, -200.0) == 0.0);

  const double src[1] = {-90.0};
  const double bound = 0.5 * sqrt(10.0);
  const double dec[1] = {-90.0 + 3.0 * bound};
  double worst = -1.0;
  CHECK(quant_sqrt_verify(src, dec, &sp, NULL, QSQ_F64, 1, &worst) == 0);
  CHECKF(fabs(worst - 3.0) < 1e-9, "worst %g against 3", worst);
}

// Topping it at the element capped a u8 grid at 255 levels, worst 57.6x.
static void an_internal_index_is_not_capped_by_the_element(void) {
  puts("an internal index is not capped by the element width");

  double field[256];
  for (int i = 0; i < 256; ++i)
    field[i] = (double)i;

  double codecWorst = -1.0;
  const double r =
      round_trip(QSQ_U8, "SQRT:MAX_ERROR=0.1N,A=0,B=1", field, 256, &codecWorst);
  CHECKF(r >= 0.0 && r <= 1.0, "worst %g", r);
  CHECKF(codecWorst >= 0.0 && codecWorst <= 1.0, "codec worst %g", codecWorst);

  // the top has to come back near itself, not folded onto the last level
  uint8_t src[256], stream[256], back[256];
  for (int i = 0; i < 256; ++i)
    src[i] = (uint8_t)i;
  quant_sqrt_spec sp;
  quant_sqrt_stats sc;
  quant_sqrt_params p;
  CHECK(quant_sqrt_parse("SQRT:MAX_ERROR=0.1N,A=0,B=1", &sp, NULL, 0) == 0);
  CHECK(quant_sqrt_scan(src, QSQ_U8, 256, &sc) == 0);
  CHECK(quant_sqrt_resolve(&sp, QSQ_U8, &sc, NULL, &p, NULL, 0) == 0);
  CHECK(quant_sqrt_encode(stream, src, &p, QSQ_U8, 256) == 0);
  CHECK(quant_sqrt_decode(back, stream, &p, QSQ_U8, 256) == 0);
  CHECKF(back[255] >= 253, "the top rebuilt as %u", back[255]);
  CHECKF(back[200] >= 198 && back[200] <= 202, "200 rebuilt as %u", back[200]);
}

// This used to resolve, encode and decode and hand back one repeated value.
static void a_degenerate_grid_is_refused_at_the_resolver(void) {
  puts("a degenerate grid is refused at the resolver");

  uint16_t src[6] = {0, 1, 10, 100, 1000, 10000};
  quant_sqrt_spec sp;
  quant_sqrt_stats sc;
  quant_sqrt_params p;
  char err[256] = {0};

  CHECK(quant_sqrt_parse("SQRT:MAX_ERROR=1e-170N,A=0,B=1", &sp, NULL, 0) == 0);
  CHECK(quant_sqrt_scan(src, QSQ_U16, 6, &sc) == 0);
  CHECK(quant_sqrt_resolve(&sp, QSQ_U16, &sc, NULL, &p, err, sizeof(err)) == 1);

  CHECK(quant_sqrt_parse("SQRT:MAX_ERROR=1e-300N,A=0,B=1", &sp, NULL, 0) == 0);
  CHECK(quant_sqrt_resolve(&sp, QSQ_U16, &sc, NULL, &p, err, sizeof(err)) == 1);
}

// Budgeting only the reconstruction let an f64 grid run 1.40x over its bound.
static void the_budget_covers_the_index_arithmetic(void) {
  puts("the budget covers the arithmetic that finds the level");

  enum { N = 4096 };
  double *field = malloc(N * sizeof(double));
  const double top = 1e18;
  for (int i = 0; i < N; ++i)
    field[i] = top * (1.0 - (double)i / (200.0 * (double)N));

  int accepted = 0;
  report_refusals = 0;
  for (double k = 1e-3; k > 1e-8; k /= 3.0) {
    char recipe[96];
    snprintf(recipe, sizeof(recipe), "SQRT:MAX_ERROR=%.17gN,A=0,B=1", k);
    double codecWorst = -1.0;
    const double r = round_trip(QSQ_F64, recipe, field, N, &codecWorst);
    if (r < 0.0)
      continue; // refused, which is the other correct answer
    ++accepted;
    CHECKF(r <= 1.0, "k=%g worst %.6f", k, r);
    CHECKF(codecWorst <= 1.0, "k=%g codec worst %.6f", k, codecWorst);
  }
  report_refusals = 1;
  CHECK(accepted > 0);
  free(field);
}

static void the_grid_holds_on_a_photon_raster(void) {
  puts("the grid holds its bound on a photon raster");

  static const struct {
    int dtype;
    double top;
  } types[] = {
      {QSQ_U8, 200.0},   {QSQ_U16, 4000.0}, {QSQ_U32, 4000.0},
      {QSQ_U64, 4000.0}, {QSQ_I8, 100.0},   {QSQ_I16, 4000.0},
      {QSQ_I32, 4000.0}, {QSQ_I64, 4000.0}, {QSQ_F16, 1800.0},
      {QSQ_F32, 4000.0}, {QSQ_F64, 4000.0},
  };
  static const char *recipes[] = {
      "SQRT:MAX_ERROR=0.5N,A=100,B=1",
      "SQRT:MAX_ERROR=1N,A=100,B=1",
      "SQRT:MAX_ERROR=2N,A=25,B=2",
      "SQRT:MAX_ERROR=1N,A=100,B=1,STORE=VALUES",
  };

  enum { W = 96, H = 96, N = W * H };
  double *field = malloc(N * sizeof(double));

  for (size_t t = 0; t < sizeof(types) / sizeof(types[0]); ++t) {
    rng_state = 0x9e3779b97f4a7c15ull; // same scene for every type
    photon_field(field, W, H, types[t].top, 100.0, 1.0);
    for (size_t r = 0; r < sizeof(recipes) / sizeof(recipes[0]); ++r) {
      double codecWorst = -1.0;
      const double worst =
          round_trip(types[t].dtype, recipes[r], field, N, &codecWorst);
      if (worst < 0.0)
        continue; // a refusal is reported by round_trip and is not a failure
      CHECKF(worst <= 1.0, "%s %s worst %.6f", dtype_name(types[t].dtype),
             recipes[r], worst);
      // the codec's measure and an independent one have to agree
      CHECKF(fabs(worst - codecWorst) < 1e-9, "%s %s, %.9f against %.9f",
             dtype_name(types[t].dtype), recipes[r], worst, codecWorst);
    }
  }
  free(field);
}

static void the_scan_reports_what_the_resolver_needs(void) {
  puts("the scan reports what the resolver needs");

  quant_sqrt_stats sc;
  const float clean[4] = {1.0f, 5.0f, 2.0f, 9.0f};
  CHECK(quant_sqrt_scan(clean, QSQ_F32, 4, &sc) == 0);
  CHECK(sc.lo == 1.0 && sc.hi == 9.0);
  CHECK(sc.anyNegative == 0 && sc.anyNonFinite == 0);

  const float signed_[4] = {-3.0f, 5.0f, 2.0f, 9.0f};
  CHECK(quant_sqrt_scan(signed_, QSQ_F32, 4, &sc) == 0);
  CHECK(sc.lo == -3.0 && sc.anyNegative == 1);

  const float withNan[4] = {1.0f, NAN, 2.0f, INFINITY};
  CHECK(quant_sqrt_scan(withNan, QSQ_F32, 4, &sc) == 0);
  CHECK(sc.lo == 1.0 && sc.hi == 2.0 && sc.anyNonFinite == 1);

  const float allNan[2] = {NAN, NAN};
  CHECK(quant_sqrt_scan(allNan, QSQ_F32, 2, &sc) == 1);

  CHECK(quant_sqrt_scan(clean, 99, 4, &sc) == 1);
  CHECK(quant_sqrt_scan(clean, QSQ_F32, 4, NULL) == 1);
}

static void the_resolver_refuses_what_the_curve_cannot_cover(void) {
  puts("the resolver refuses what the curve cannot cover");

  quant_sqrt_spec sp;
  quant_sqrt_stats sc;
  quant_sqrt_params p;
  char err[256] = {0};
  CHECK(quant_sqrt_parse("SQRT:MAX_ERROR=0.5N,A=100,B=1", &sp, NULL, 0) == 0);

  const double ok[3] = {-99.0, 0.0, 500.0};
  CHECK(quant_sqrt_scan(ok, QSQ_F64, 3, &sc) == 0);
  CHECK(quant_sqrt_resolve(&sp, QSQ_F64, &sc, NULL, &p, err, sizeof(err)) == 0);
  CHECK((p.flags & QUANT_SQRT_FLAG_NONNEGATIVE) == 0);

  const double past[3] = {-101.0, 0.0, 500.0};
  CHECK(quant_sqrt_scan(past, QSQ_F64, 3, &sc) == 0);
  CHECK(quant_sqrt_resolve(&sp, QSQ_F64, &sc, NULL, &p, err, sizeof(err)) == 1);

  const double nonneg[3] = {0.0, 10.0, 500.0};
  CHECK(quant_sqrt_scan(nonneg, QSQ_F64, 3, &sc) == 0);
  CHECK(quant_sqrt_resolve(&sp, QSQ_F64, &sc, NULL, &p, err, sizeof(err)) == 0);
  CHECK((p.flags & QUANT_SQRT_FLAG_NONNEGATIVE) != 0);

  // no A and no B, and no fit to fall back on
  CHECK(quant_sqrt_parse("SQRT:MAX_ERROR=0.5N", &sp, NULL, 0) == 0);
  CHECK(quant_sqrt_resolve(&sp, QSQ_F64, &sc, NULL, &p, err, sizeof(err)) == 1);
}

// Two tiles can each hold the bound and still disagree by twice it, and no per
// sample check sees that. Found by the fuzzer.
static void the_value_grid_does_not_move_with_the_tile(void) {
  puts("the value grid does not move with the tile");

  static const char *recipes[] = {"SQRT:MAX_ERROR=1N,A=100,B=1,STORE=VALUES",
                                  "SQRT:MAX_ERROR=0.5N,A=25,B=2,STORE=VALUES",
                                  "SQRT:MAX_ERROR=2N,A=0,B=1,STORE=VALUES"};
  static const int types[] = {QSQ_U8, QSQ_U16, QSQ_I16, QSQ_U32,
                              QSQ_F32, QSQ_F64};

  enum { N = 512 };
  double dim[N], bright[N];
  for (int i = 0; i < N; ++i) {
    dim[i] = 20.0 + 0.05 * i;   // one tile in shadow
    bright[i] = 90.0 + 0.2 * i; // and one in sun
  }

  for (size_t r = 0; r < sizeof(recipes) / sizeof(recipes[0]); ++r) {
    quant_sqrt_spec sp;
    CHECK(quant_sqrt_parse(recipes[r], &sp, NULL, 0) == 0);
    for (size_t t = 0; t < sizeof(types) / sizeof(types[0]); ++t) {
      const int dt = types[t];
      const size_t w = quant_sqrt_width(dt);
      void *a = malloc(N * w), *b = malloc(N * w);
      store_as(a, dt, dim, N);
      store_as(b, dt, bright, N);

      quant_sqrt_stats sa, sb;
      quant_sqrt_params pa, pb;
      if (quant_sqrt_scan(a, dt, N, &sa) == 0 &&
          quant_sqrt_scan(b, dt, N, &sb) == 0 &&
          quant_sqrt_resolve(&sp, dt, &sa, NULL, &pa, NULL, 0) == 0 &&
          quant_sqrt_resolve(&sp, dt, &sb, NULL, &pb, NULL, 0) == 0) {
        CHECKF(pa.step == pb.step, "%s %s, %.17g against %.17g", dtype_name(dt),
               recipes[r], pa.step, pb.step);
        CHECK(pa.offset == pb.offset);
      }
      free(a);
      free(b);
    }
  }
}

// Near -offset the bound is small and the magnitude is not. Charging at maxAbs
// paired with the u of the other end let an f16 grid run over. Found by the
// fuzzer.
static void the_charge_is_read_at_the_worst_end(void) {
  puts("the reconstruction charge is read at the worst end");

  // an anomaly field, mostly negative, on a curve anchored at A/B = 100
  static const int types[] = {QSQ_F16, QSQ_F32, QSQ_F64};
  enum { N = 400 };
  double field[N];
  for (int i = 0; i < N; ++i)
    field[i] = -99.0 + 0.26 * i; // reaches from just above -offset up past zero

  for (size_t t = 0; t < sizeof(types) / sizeof(types[0]); ++t) {
    double codecWorst = -1.0;
    const double r = round_trip(types[t], "SQRT:MAX_ERROR=0.5N,A=100,B=1", field,
                                N, &codecWorst);
    if (r < 0.0)
      continue;
    CHECKF(r <= 1.0, "%s worst %.6f", dtype_name(types[t]), r);
    CHECKF(codecWorst <= 1.0, "%s codec worst %.6f", dtype_name(types[t]),
           codecWorst);
  }

  // the tile the fuzzer reduced to
  const double one[2] = {-89.5, 5.0};
  double codecWorst = -1.0;
  const double r =
      round_trip(QSQ_F16, "SQRT:MAX_ERROR=0.5N,A=100,B=1", one, 2, &codecWorst);
  CHECKF(r >= 0.0 && r <= 1.0, "worst %.6f", r);
}

static void an_empty_raster_is_not_a_crash(void) {
  puts("an empty raster is not a crash");

  quant_sqrt_params p;
  memset(&p, 0, sizeof(p));
  p.step = 0.5;
  p.offset = 0.0;
  float dummy = 0.0f;
  CHECK(quant_sqrt_encode(&dummy, &dummy, &p, QSQ_F32, 0) == 0);
  CHECK(quant_sqrt_decode(&dummy, &dummy, &p, QSQ_F32, 0) == 0);
}

// Every value of a type through the grid, one at a time. The small domains are
// walked whole; f32 goes on a stride unless GEOZL_EXHAUSTIVE is set. The curve
// is fixed in the recipe, so nothing here depends on a fit.
static void every_value_holds_the_bound(void) {
  static const char *const recipes[] = {"SQRT:MAX_ERROR=0.5N,A=100,B=1",
                                        "SQRT:MAX_ERROR=2N,A=100,B=1"};
  static const double ks[] = {0.5, 2.0};
  static const double tops[] = {255.0, 65535.0, 32767.0, 1e4, 1e4};
  static uint8_t in[GEOZL_WALK_BLK * 4], st[GEOZL_WALK_BLK * 4];
  static uint8_t out[GEOZL_WALK_BLK * 4];
  const double A = 100.0, B = 1.0;
  const uint64_t step = geozl_walk_step();
  printf("every value of a type holds the bound, f32 %s\n",
         step == 1 ? "whole" : "on a stride");

  for (size_t t = 0; t < GEOZL_WALK_NTYPES; ++t) {
    const int dt = geozl_walk_types[t].dtype;
    for (size_t r = 0; r < sizeof(ks) / sizeof(ks[0]); ++r) {
      char err[256];
      quant_sqrt_spec sp;
      quant_sqrt_stats sc;
      quant_sqrt_params p;
      memset(&sc, 0, sizeof sc);
      sc.lo = 0.0;
      sc.hi = tops[t];
      CHECK(quant_sqrt_parse(recipes[r], &sp, err, sizeof err) == 0);
      if (quant_sqrt_resolve(&sp, dt, &sc, NULL, &p, err, sizeof err) != 0) {
        printf("  %-4s %-30s refused, %s\n", geozl_walk_types[t].name,
               recipes[r], err);
        continue;
      }

      double worst = 0.0;
      uint64_t counted = 0, over = 0, sum = 0, at = 0;
      const uint64_t domain = geozl_walk_types[t].domain;
      const uint64_t span = dt == GEOZL_DT_F32 ? step : 1;
      for (uint64_t base = 0; base < domain; base += GEOZL_WALK_BLK * span) {
        const size_t n = geozl_walk_fill(in, dt, base, domain, span);
        CHECK(quant_sqrt_encode(st, in, &p, dt, n) == 0);
        CHECK(quant_sqrt_decode(out, st, &p, dt, n) == 0);
        for (size_t i = 0; i < n; ++i) {
          const uint64_t pos = base + (uint64_t)i * span;
          GEOZL_FOLD(sum, geozl_walk_bits(out, dt, i), pos);
          const double x = geozl_walk_get(in, dt, i);
          const double y = geozl_walk_get(out, dt, i);
          if (!isfinite(x))
            continue; // the nodata codec owns these
          // The curve is only defined at or above -A/B, and the grid was cut
          // for a range a raster holds rather than for the whole domain.
          if (x < 0.0 || x > tops[t])
            continue;
          ++counted;
          const double bound = ks[r] * sqrt(A + B * x);
          const double e = fabs(y - x);
          if (e / bound > worst) {
            worst = e / bound;
            at = pos;
          }
          if (e > bound)
            ++over;
        }
      }
      printf("  %-4s %-30s %10llu values, worst %.4f of the bound, sum %016llx\n",
             geozl_walk_types[t].name, recipes[r], (unsigned long long)counted,
             worst, (unsigned long long)sum);
      if (over != 0) {
        printf("    FAIL %llu over the bound, worst at 0x%llx\n",
               (unsigned long long)over, (unsigned long long)at);
        ++failures;
      }
    }
  }
}

int main(void) {
  the_parser_takes_only_valid_recipes();
  a_comma_locale_reads_the_same_recipe();
  the_curve_is_read_at_signed_x();
  an_internal_index_is_not_capped_by_the_element();
  a_degenerate_grid_is_refused_at_the_resolver();
  the_budget_covers_the_index_arithmetic();
  the_grid_holds_on_a_photon_raster();
  the_scan_reports_what_the_resolver_needs();
  the_resolver_refuses_what_the_curve_cannot_cover();
  the_value_grid_does_not_move_with_the_tile();
  the_charge_is_read_at_the_worst_end();
  an_empty_raster_is_not_a_crash();
  every_value_holds_the_bound();

  if (failures != 0) {
    printf("%d failed\n", failures);
    return 1;
  }
  puts("all passed");
  return 0;
}