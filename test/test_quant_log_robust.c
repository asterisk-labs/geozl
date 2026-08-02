// What the two ends agree to refuse, what the parser takes, and the edges of a
// signed type. Every case here was a real hole.

#include "quant_log/decode_quant_log_kernel.h"
#include "quant_log/encode_quant_log_kernel.h"
#include "quant_log/quant_log_dtype.h"
#include "quant_log/quant_log_spec.h"

#include <locale.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
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
      {"an index grid on float32", QLOG_F32, 0, 0.0144, 1},
      {"a value grid on float32", QLOG_F32, QUANT_LOG_FLAG_STORE_VALUES, 0.0144,
       1},
      {"a value grid on uint16", QLOG_U16, QUANT_LOG_FLAG_STORE_VALUES, 0.0144,
       1},
      {"an index grid on uint16", QLOG_U16, 0, 0.0144, 0},
      {"a flag bit nothing defines", QLOG_F32, 4, 0.0144, 0},
      {"a step of zero", QLOG_F32, 0, 0.0, 0},
      {"a negative step", QLOG_F32, 0, -0.0144, 0},
      {"an infinite step", QLOG_F32, 0, INFINITY, 0},
      {"a nan step", QLOG_F32, 0, NAN, 0},
      {"a type past the table", QLOG_F64 + 1, QUANT_LOG_FLAG_STORE_VALUES,
       0.0144, 0},
  };

  unsigned char in[64], out[64];
  memset(in, 0, sizeof(in));
  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    quant_log_params p;
    p.flags = cases[i].flags;
    p.step = cases[i].step;
    const int e = quant_log_encode(out, in, &p, cases[i].dtype, 4) == 0;
    const int d = quant_log_decode(out, in, &p, cases[i].dtype, 4) == 0;
    if (e != cases[i].ok || d != cases[i].ok)
      printf("  %s: encode %d decode %d, expected %d\n", cases[i].what, e, d,
             cases[i].ok);
    CHECK(e == cases[i].ok);
    CHECK(d == cases[i].ok);
  }
}

static void the_parser_takes_only_decimals(void) {
  puts("the parser takes only decimals");

  static const struct {
    const char *recipe;
    int ok;
  } cases[] = {
      {"LOG:MAX_ERROR=1%", 1},
      {"LOG:MAX_ERROR=0.5%", 1},
      {"LOG:MAX_ERROR=1e-3%", 1},
      {"LOG:MAX_ERROR=1E+1%", 1},
      {"LOG:MAX_ERROR=.5%", 1},
      {"LOG:MAX_ERROR=1%,STORE=VALUES", 1},
      {"LOG:MAX_ERROR=1%,STORE=INDEX", 1},
      {"LOG:MAX_ERROR=1", 0},   // a bound of one is not what anybody means
      {"LOG:MAX_ERROR=0%", 0},
      {"LOG:MAX_ERROR=100%", 0},
      {"LOG:MAX_ERROR=-1%", 0},
      {"LOG:MAX_ERROR=inf%", 0},
      {"LOG:MAX_ERROR=nan%", 0},
      {"LOG:MAX_ERROR=1.0.0%", 0},
      {"LOG:MAX_ERROR=1e%", 0},
      {"LOG:MAX_ERROR=1_0%", 0},
      {"LOG:MAX_ERROR=1e999%", 0},
      {"LOG:MAX_ERROR=1% ", 0},
      {"LOG:MAX_ERROR=1%,MAX_ERROR=2%", 0},
      {"LOG:MAX_ERROR=1%,", 0},
      {"LOG:STORE=VALUES", 0},
      {"LOG:", 0},
      {"LINEAR:MAX_ERROR=1", 0},
  };

  char err[256];
  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    quant_log_spec sp;
    const int got = quant_log_parse(cases[i].recipe, &sp, err, sizeof(err)) == 0;
    if (got != cases[i].ok)
      printf("  \"%s\": got %d, expected %d\n", cases[i].recipe, got,
             cases[i].ok);
    CHECK(got == cases[i].ok);
  }
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

  quant_log_spec sp;
  char err[256];
  CHECK(quant_log_parse("LOG:MAX_ERROR=0.5%", &sp, err, sizeof(err)) == 0);
  CHECK(sp.rel_err == 0.005);
  // The comma separates keys, so it never reaches the number.
  CHECK(quant_log_parse("LOG:MAX_ERROR=0,5%", &sp, err, sizeof(err)) != 0);

  setlocale(LC_NUMERIC, "C");
}

// Found by the fuzzer on int16 at 0.002%. Folding the grid onto the maximum
// rather than the magnitude brought the most negative value back one short.
static void the_most_negative_value_survives(void) {
  puts("the most negative value of a signed type survives");

  static const struct {
    const char *what;
    int dtype;
    int64_t lowest;
  } cases[] = {
      {"int8", QLOG_I8, INT8_MIN},
      {"int16", QLOG_I16, INT16_MIN},
      {"int32", QLOG_I32, INT32_MIN},
      {"int64", QLOG_I64, INT64_MIN},
  };
  static const char *recipes[] = {"LOG:MAX_ERROR=0.002%", "LOG:MAX_ERROR=1%",
                                  "LOG:MAX_ERROR=5%"};

  for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); ++c) {
    for (size_t r = 0; r < sizeof(recipes) / sizeof(recipes[0]); ++r) {
      int64_t in[4];
      in[0] = cases[c].lowest;
      in[1] = cases[c].lowest + 1;
      in[2] = -1;
      in[3] = 1;

      unsigned char src[4 * 8], enc[4 * 8], dec[4 * 8];
      const size_t w = quant_log_width(cases[c].dtype);
      for (size_t i = 0; i < 4; ++i)
        memcpy(src + i * w, &in[i], w); // little endian, which the tests assume

      char err[256];
      quant_log_spec sp;
      quant_log_params p;
      quant_log_stats sc;
      CHECK(quant_log_parse(recipes[r], &sp, err, sizeof(err)) == 0);
      CHECK(quant_log_scan(src, cases[c].dtype, 4, &sc) == 0);
      if (quant_log_resolve(&sp, cases[c].dtype, &sc, &p, err, sizeof(err)) != 0)
        continue;
      CHECK(quant_log_encode(enc, src, &p, cases[c].dtype, 4) == 0);
      CHECK(quant_log_decode(dec, enc, &p, cases[c].dtype, 4) == 0);

      double worst = 0.0;
      size_t skipped = 0;
      CHECK(quant_log_verify(src, dec, &sp, cases[c].dtype, 4, &worst,
                             &skipped) == 0);
      if (worst > 1.0)
        printf("  %s %s: worst %.6f of the bound\n", cases[c].what, recipes[r],
               worst);
      CHECK(worst <= 1.0);
    }
  }
}

// A frame that declares the floor has to get it whatever the element type is.
// The float paths always applied it and the integer copy did not, so the flag
// meant two things inside one codec.
static void the_floor_reaches_an_integer_frame(void) {
  puts("the floor reaches an integer frame");

  const int16_t stream[6] = {-3000, -1, 0, 1, 500, 32767};
  int16_t out[6];
  quant_log_params p = {QUANT_LOG_FLAG_NONNEGATIVE | QUANT_LOG_FLAG_STORE_VALUES,
                        0.0144};
  CHECK(quant_log_decode(out, stream, &p, QLOG_I16, 6) == 0);
  for (size_t i = 0; i < 6; ++i)
    CHECK(out[i] >= 0);
  CHECK(out[3] == 1 && out[4] == 500 && out[5] == 32767);

  // Without the flag the same stream comes back untouched.
  p.flags = QUANT_LOG_FLAG_STORE_VALUES;
  CHECK(quant_log_decode(out, stream, &p, QLOG_I16, 6) == 0);
  CHECK(out[0] == -3000 && out[1] == -1);

  // An unsigned type cannot hold a negative, so the flag changes nothing there.
  const uint16_t ustream[4] = {0, 1, 500, 65535};
  uint16_t uout[4];
  p.flags = QUANT_LOG_FLAG_NONNEGATIVE | QUANT_LOG_FLAG_STORE_VALUES;
  CHECK(quant_log_decode(uout, ustream, &p, QLOG_U16, 4) == 0);
  CHECK(memcmp(uout, ustream, sizeof(ustream)) == 0);
}

int main(void) {
  the_two_ends_refuse_the_same_blocks();
  the_parser_takes_only_decimals();
  a_comma_locale_reads_the_same_recipe();
  the_most_negative_value_survives();
  the_floor_reaches_an_integer_frame();

  if (failures != 0) {
    printf("\n%d failed\n", failures);
    return 1;
  }
  puts("\nall passed");
  return 0;
}
