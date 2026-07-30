// Beyond the bound holding, what this pins down is which paths read the tile.
// A grid that follows the tile makes the same value reconstruct two ways once the
// raster is cut differently, and no per-sample check on the error sees it.

#include "quant_linear/decode_quant_linear_kernel.h"
#include "quant_linear/encode_quant_linear_kernel.h"
#include "quant_linear/quant_linear_dtype.h"
#include "quant_linear/quant_linear_half.h"
#include "quant_linear/quant_linear_spec.h"

#include <math.h>
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

// One trip the way the encode binding does it, so the step this measures is the
// step a frame would be written with.
static int trip(const char *recipe, int dtype, const void *src, void *dec,
                size_t n, quant_linear_params *pOut) {
  char err[256];
  quant_linear_spec sp;
  if (quant_linear_parse(recipe, &sp, err, sizeof(err)) != 0)
    return -1;
  double hi = 0.0;
  int neg = 0;
  quant_linear_scan(src, dtype, n, &hi, &neg);
  quant_linear_params p;
  if (quant_linear_resolve(&sp, dtype, hi, neg, &p, err, sizeof(err)) != 0)
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

static double get(const void *p, int dtype, size_t i) {
  switch (dtype) {
  case QL_U8:
    return ((const uint8_t *)p)[i];
  case QL_U16:
    return ((const uint16_t *)p)[i];
  case QL_U32:
    return ((const uint32_t *)p)[i];
  case QL_U64:
    return (double)((const uint64_t *)p)[i];
  case QL_I8:
    return ((const int8_t *)p)[i];
  case QL_I16:
    return ((const int16_t *)p)[i];
  case QL_I32:
    return ((const int32_t *)p)[i];
  case QL_I64:
    return (double)((const int64_t *)p)[i];
  case QL_F16:
    return quant_linear_half_to_float(((const uint16_t *)p)[i]);
  case QL_F32:
    return ((const float *)p)[i];
  default:
    return ((const double *)p)[i];
  }
}

static void put(void *p, int dtype, size_t i, double v) {
  switch (dtype) {
  case QL_U8:
    ((uint8_t *)p)[i] = (uint8_t)v;
    break;
  case QL_U16:
    ((uint16_t *)p)[i] = (uint16_t)v;
    break;
  case QL_U32:
    ((uint32_t *)p)[i] = (uint32_t)v;
    break;
  case QL_U64:
    ((uint64_t *)p)[i] = (uint64_t)v;
    break;
  case QL_I8:
    ((int8_t *)p)[i] = (int8_t)v;
    break;
  case QL_I16:
    ((int16_t *)p)[i] = (int16_t)v;
    break;
  case QL_I32:
    ((int32_t *)p)[i] = (int32_t)v;
    break;
  case QL_I64:
    ((int64_t *)p)[i] = (int64_t)v;
    break;
  case QL_F16:
    ((uint16_t *)p)[i] = quant_linear_float_to_half((float)v);
    break;
  case QL_F32:
    ((float *)p)[i] = (float)v;
    break;
  default:
    ((double *)p)[i] = v;
    break;
  }
}

static void the_bound_holds(void) {
  printf("the declared bound holds, every type\n");
  enum { N = 4096 };
  // Bounds that are not whole numbers on purpose: 0.94 asks for a step of 1.88,
  // and rounding that up to 2 would miss by 1.
  static const double bounds[] = {0.25, 0.94, 1.0, 5.0, 12.75, 50.0};
  static const int dts[] = {QL_U8,  QL_U16, QL_U32, QL_U64, QL_I8, QL_I16,
                            QL_I32, QL_I64, QL_F16, QL_F32, QL_F64};

  for (size_t d = 0; d < sizeof(dts) / sizeof(*dts); ++d) {
    const int dt = dts[d];
    const size_t w = quant_linear_width(dt);
    const int signd = (dt >= QL_I8);
    void *src = calloc(N, w);
    void *dec = calloc(N, w);
    CHECK(src != NULL && dec != NULL);
    for (size_t i = 0; i < N; ++i) {
      const double v = (double)(i % 100);
      put(src, dt, i, signd ? v - 50.0 : v);
    }
    for (size_t b = 0; b < sizeof(bounds) / sizeof(*bounds); ++b) {
      char recipe[48];
      snprintf(recipe, sizeof(recipe), "LINEAR:MAX_ERROR=%g", bounds[b]);
      if (trip(recipe, dt, src, dec, N, NULL) != 0)
        continue; // a refusal is an answer
      for (size_t i = 0; i < N; ++i) {
        const double x = get(src, dt, i), y = get(dec, dt, i);
        if (fabs(x - y) > bounds[b]) {
          printf("  %s dtype %d sample %zu: |%g - %g| = %g > %g\n", recipe, dt,
                 i, x, y, fabs(x - y), bounds[b]);
          ++failures;
          break;
        }
      }
    }
    free(src);
    free(dec);
  }
}

static void an_integer_grid_never_reads_the_tile(void) {
  printf("an integer grid is the same grid whatever the tile holds\n");
  char err[256];
  quant_linear_spec sp;
  quant_linear_params a, b, c;
  CHECK(quant_linear_parse("LINEAR:MAX_ERROR=5", &sp, err, sizeof(err)) == 0);

  static const int dts[] = {QL_U8, QL_U16, QL_I16, QL_I32};
  for (size_t d = 0; d < sizeof(dts) / sizeof(*dts); ++d) {
    CHECK(quant_linear_resolve(&sp, dts[d], 3.0, 0, &a, err,
                               sizeof(err)) == 0);
    CHECK(quant_linear_resolve(&sp, dts[d], 200.0, 0, &b, err,
                               sizeof(err)) == 0);
    CHECK(quant_linear_resolve(&sp, dts[d], 100.0, 1, &c, err,
                               sizeof(err)) == 0);
    CHECK(a.step == b.step);
    CHECK(a.step == c.step);
    CHECK(a.step == 10.0);
    CHECK((a.flags & QUANT_LINEAR_FLAG_STORE_VALUES) != 0);
    CHECK((a.flags & QUANT_LINEAR_FLAG_NONNEGATIVE) != 0);
    CHECK((c.flags & QUANT_LINEAR_FLAG_NONNEGATIVE) == 0);
  }

  // The step never rounds up past the bound.
  CHECK(quant_linear_parse("LINEAR:MAX_ERROR=0.94", &sp, err, sizeof(err)) == 0);
  CHECK(quant_linear_resolve(&sp, QL_U16, 1000.0, 0, &a, err,
                             sizeof(err)) == 0);
  CHECK(a.step == 1.0); // floor(1.88), not 2
  CHECK(quant_linear_parse("LINEAR:MAX_ERROR=12.75", &sp, err, sizeof(err)) == 0);
  CHECK(quant_linear_resolve(&sp, QL_I16, 1000.0, 1, &a, err,
                             sizeof(err)) == 0);
  CHECK(a.step == 25.0); // floor(25.5), not 26
  // And a bound under half a unit is lossless rather than a step of zero.
  CHECK(quant_linear_parse("LINEAR:MAX_ERROR=0.4", &sp, err, sizeof(err)) == 0);
  CHECK(quant_linear_resolve(&sp, QL_U8, 200.0, 0, &a, err,
                             sizeof(err)) == 0);
  CHECK(a.step == 1.0);
}

static void the_float_index_path_reads_the_tile(void) {
  printf("a float index grid follows the tile\n");
  char err[256];
  quant_linear_spec sp;
  quant_linear_params a, b;
  CHECK(quant_linear_parse("LINEAR:MAX_ERROR=0.05", &sp, err, sizeof(err)) == 0);
  CHECK(sp.store == QUANT_LINEAR_STORE_INDEX);
  CHECK(quant_linear_resolve(&sp, QL_F32, 100.0, 0, &a, err,
                             sizeof(err)) == 0);
  CHECK(quant_linear_resolve(&sp, QL_F32, 300.0, 0, &b, err,
                             sizeof(err)) == 0);
  CHECK(a.step != b.step); // the documented limitation of this path
  CHECK((a.flags & QUANT_LINEAR_FLAG_STORE_VALUES) == 0);

  // A bound below what the output type resolves is refused rather than silently
  // returning the data untouched. At a million a float32 only resolves 0.0625.
  CHECK(quant_linear_parse("LINEAR:MAX_ERROR=1e-5", &sp, err, sizeof(err)) == 0);
  CHECK(quant_linear_resolve(&sp, QL_F32, 1e5, 0, &a, err, sizeof(err)) !=
        0);
}

static void the_float_value_path_does_not(void) {
  printf("a float value grid does not, and is refused rather than falling back\n");
  char err[256];
  quant_linear_spec sp;
  quant_linear_params a, b;
  CHECK(quant_linear_parse("LINEAR:MAX_ERROR=0.5,STORE=VALUES", &sp, err,
                           sizeof(err)) == 0);
  CHECK(sp.store == QUANT_LINEAR_STORE_VALUES);
  CHECK(quant_linear_resolve(&sp, QL_F32, 100.0, 0, &a, err,
                             sizeof(err)) == 0);
  CHECK(quant_linear_resolve(&sp, QL_F32, 30000.0, 0, &b, err,
                             sizeof(err)) == 0);
  CHECK(a.step == b.step);
  CHECK(a.step == 1.0);
  CHECK((a.flags & QUANT_LINEAR_FLAG_STORE_VALUES) != 0);

  // A bound too tight for a whole step. This is why it cannot be automatic:
  // reflectance in 0 to 1 never gets a whole step, and a fallback would put the
  // dependence on the tile back in.
  quant_linear_spec fine;
  CHECK(quant_linear_parse("LINEAR:MAX_ERROR=0.0005,STORE=VALUES", &fine, err,
                           sizeof(err)) == 0);
  CHECK(quant_linear_resolve(&fine, QL_F32, 1.0, 0, &a, err,
                             sizeof(err)) != 0);
  // Past 2^24 a float32 no longer holds every integer, so the cast back would
  // round.
  CHECK(quant_linear_resolve(&sp, QL_F32, 3e7, 0, &a, err, sizeof(err)) !=
        0);
  // On an integer type the reconstruction is what the stream always carries, so
  // asking for it changes nothing.
  CHECK(quant_linear_resolve(&sp, QL_U16, 60000.0, 0, &a, err,
                             sizeof(err)) == 0);
  CHECK(a.step == 1.0);
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

  // An integer frame without the flag is one this codec never writes.
  p.step = 1.0;
  p.flags = 0;
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

int main(void) {
  the_bound_holds();
  an_integer_grid_never_reads_the_tile();
  the_float_index_path_reads_the_tile();
  the_float_value_path_does_not();
  a_float_array_of_integers_round_trips_exactly();
  non_negative_stays_non_negative();
  the_parser_is_strict();
  a_forged_header_is_refused();
  the_edges();
  if (failures != 0) {
    printf("%d failed\n", failures);
    return 1;
  }
  printf("all passed\n");
  return 0;
}
