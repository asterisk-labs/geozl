// What the two ends agree to refuse, and what the parser takes. Every case here
// was a real hole.

#include "quant_linear/decode_quant_linear_kernel.h"
#include "quant_linear/encode_quant_linear_kernel.h"
#include "quant_linear/quant_linear_dtype.h"
#include "quant_linear/quant_linear_spec.h"

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

// Either both ends take a block or neither does, or one side can build what the
// other cannot read.
static void the_two_ends_refuse_the_same_blocks(void) {
  puts("the two ends refuse the same parameter blocks");

  static const struct {
    const char *what;
    int dtype;
    unsigned char flags;
    double step;
    int ok;
  } cases[] = {
      {"a whole step on float32", QL_F32, QUANT_LINEAR_FLAG_STORE_VALUES, 2.0, 1},
      {"an index grid on float32", QL_F32, 0, 0.25, 1},
      {"an integer grid", QL_U16, QUANT_LINEAR_FLAG_STORE_VALUES, 10.0, 1},
      {"an infinite step", QL_F32, 0, INFINITY, 0},
      {"a nan step", QL_F32, 0, NAN, 0},
      {"a zero step", QL_F32, 0, 0.0, 0},
      {"a negative step", QL_F32, 0, -1.0, 0},
      {"a subnormal step", QL_F32, 0, 5e-324, 0},
      {"the smallest normal step", QL_F32, 0, DBL_MIN, 1},
      {"an unknown flag bit", QL_F32, 4, 1.0, 0},
      {"an integer grid without VALUES", QL_U16, 0, 10.0, 0},
      {"a fractional step under VALUES", QL_F32,
       QUANT_LINEAR_FLAG_STORE_VALUES, 2.5, 0},
      {"a dtype past the table", 99, QUANT_LINEAR_FLAG_STORE_VALUES, 1.0, 0},
  };

  enum { N = 8 };
  float src[N];
  int32_t stream[N];
  float back[N];
  for (int i = 0; i < N; ++i)
    src[i] = (float)(i + 1);

  for (size_t k = 0; k < sizeof(cases) / sizeof(cases[0]); ++k) {
    quant_linear_params p;
    memset(&p, 0, sizeof(p));
    p.flags = cases[k].flags;
    p.step = cases[k].step;

    memset(stream, 0, sizeof(stream));
    const int e = quant_linear_encode(stream, src, &p, cases[k].dtype, N);
    const int d = quant_linear_decode(back, stream, &p, cases[k].dtype, N);
    if ((e == 0) != (d == 0))
      printf("  FAIL %s: encode %d, decode %d\n", cases[k].what, e, d);
    CHECK((e == 0) == (d == 0));
    CHECK((e == 0) == (cases[k].ok != 0));
  }
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
    double hi = 0.0, worst = -1.0;
    int neg = 0;
    CHECK(quant_linear_parse(cases[k].recipe, &sp, err, sizeof(err)) == 0);
    quant_linear_scan(src, cases[k].dtype, N, &hi, &neg);
    CHECK(quant_linear_resolve(&sp, cases[k].dtype, hi, neg, &p, err,
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

int main(void) {
  the_two_ends_refuse_the_same_blocks();
  the_parser_takes_only_decimals();
  a_comma_locale_reads_the_same_recipe();
  verify_sees_the_bound_hold();
  verify_sees_the_bound_break();
  if (failures != 0) {
    printf("%d failed\n", failures);
    return 1;
  }
  puts("all passed");
  return 0;
}
