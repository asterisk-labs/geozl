// The bound over every value of a type, one at a time. Minutes rather than
// seconds, so it is kept out of the fast test run.
//
// Three claims live here. That every value of u8, u16, i16 and f16 round trips
// inside the bound. That every normal f32 does. And that the subnormals do not,
// which is the limit the spec states, so a change that quietly widened the grid
// to cover them would fail here rather than pass unnoticed.
//
// Each line also carries a checksum of every reconstruction it made. Matching
// worst cases across two builds says the statistics agree; matching checksums
// says all four billion values came back with the same bits. Compare them across
// compilers, flags and platforms.

#include "quant_log/decode_quant_log_kernel.h"
#include "quant_log/encode_quant_log_kernel.h"
#include "quant_log/quant_log_dtype.h"
#include "quant_log/quant_log_half.h"
#include "quant_log/quant_log_spec.h"

#include <inttypes.h>
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

// Every reconstruction folded into one number, the same way
// test_quant_exhaustive.c does it. Addition, so the order it is fed in cannot
// change the result.
#define FOLD(sum, bits, pos)                                                   \
  ((sum) += (uint64_t)(bits) * 1099511628211ull + (uint64_t)(pos))

static const char *const recipes[] = {"LOG:MAX_ERROR=5%", "LOG:MAX_ERROR=1%",
                                      "LOG:MAX_ERROR=0.1%"};
static const double bounds[] = {0.05, 0.01, 0.001};
#define NREC 3

// anyNegative is set so the decoder floors at the type minimum rather than at
// zero, which puts the negative half of the domain through the grid instead of
// through the clamp.
static int setup(const char *recipe, int dtype, quant_log_params *p) {
  char err[256];
  quant_log_spec sp;
  quant_log_stats sc;
  memset(&sc, 0, sizeof sc);
  sc.anyNegative = 1;
  sc.minAbs = 1.0;
  sc.maxAbs = 1.0;
  if (quant_log_parse(recipe, &sp, err, sizeof err) != 0)
    return 1;
  if (quant_log_resolve(&sp, dtype, &sc, p, err, sizeof err) != 0) {
    printf("  %-20s dtype %d: %s\n", recipe, dtype, err);
    return 1;
  }
  return 0;
}

#define EXHAUSTIVE_INT(NAME, T, UT, DT, LO, N)                                 \
  static void NAME(void) {                                                     \
    static T in[N], st[N], out[N];                                             \
    for (long i = 0; i < (N); ++i)                                             \
      in[i] = (T)((LO) + i);                                                   \
    for (int r = 0; r < NREC; ++r) {                                           \
      quant_log_params p;                                                      \
      if (setup(recipes[r], DT, &p) != 0)                                      \
        continue;                                                              \
      CHECK(quant_log_encode(st, in, &p, DT, N) == 0);                         \
      CHECK(quant_log_decode(out, st, &p, DT, N) == 0);                        \
      double worst = 0.0;                                                      \
      long over = 0, exact = 0;                                                \
      double firstMoved = 0.0;                                                 \
      uint64_t sum = 0;                                                        \
      for (long i = 0; i < (N); ++i) {                                         \
        FOLD(sum, (UT)out[i], i);                                              \
        const double x = (double)in[i], y = (double)out[i];                    \
        if (x == 0.0) {                                                        \
          CHECK(y == 0.0);                                                     \
          continue;                                                            \
        }                                                                      \
        if (x == y) {                                                          \
          ++exact;                                                             \
          continue;                                                            \
        }                                                                      \
        if (firstMoved == 0.0 && x > 0.0)                                      \
          firstMoved = x;                                                      \
        const double e = fabs(y - x) / fabs(x);                                \
        if (e > worst)                                                         \
          worst = e;                                                           \
        if (e > bounds[r])                                                     \
          ++over;                                                              \
      }                                                                        \
      printf("  %-8s %-20s worst %.6f, over %ld, exact %ld, first to move "   \
             "%.0f, sum %016" PRIx64 "\n",                                     \
             #T, recipes[r], worst, over, exact, firstMoved, sum);             \
      CHECK(over == 0);                                                        \
    }                                                                          \
  }

EXHAUSTIVE_INT(every_u8, uint8_t, uint8_t, QLOG_U8, 0, 256)
EXHAUSTIVE_INT(every_u16, uint16_t, uint16_t, QLOG_U16, 0, 65536)
EXHAUSTIVE_INT(every_i16, int16_t, uint16_t, QLOG_I16, -32768, 65536)

// Half the distance to the next value the type represents, relative, read off
// the bit pattern so it is the spacing of a half and not of the float carrying
// it. On a normal this is the eps the resolver budgets for. On a subnormal it
// grows without limit as the value falls, which is why the bound is stated from
// the smallest normal up.
static double half_ulp_rel16(uint16_t h) {
  const uint16_t a = (uint16_t)(h & 0x7FFFu);
  const double x = quant_log_half_to_float(a);
  const double nx = quant_log_half_to_float((uint16_t)(a + 1u));
  return 0.5 * (nx - x) / x;
}

static void every_half(void) {
  static uint16_t in[65536], st[65536], out[65536];
  for (int i = 0; i < 65536; ++i)
    in[i] = (uint16_t)i;
  const float nmin = (float)quant_log_normal_min(QLOG_F16);

  for (int r = 0; r < NREC; ++r) {
    quant_log_params p;
    if (setup(recipes[r], QLOG_F16, &p) != 0)
      continue;
    CHECK(quant_log_encode(st, in, &p, QLOG_F16, 65536) == 0);
    CHECK(quant_log_decode(out, st, &p, QLOG_F16, 65536) == 0);
    double wN = 0.0, wS = 0.0;
    int overN = 0, overS = 0, nsub = 0;
    uint64_t sum = 0;
    for (int i = 0; i < 65536; ++i) {
      FOLD(sum, out[i], i);
      const float x = quant_log_half_to_float(in[i]);
      const float y = quant_log_half_to_float(out[i]);
      if (!isfinite(x) || x == 0.0f)
        continue;
      const double e = fabs((double)y - (double)x) / fabs((double)x);
      // The claim that holds everywhere. Nothing lands further off than the
      // bound plus the rounding of the type at that magnitude, so an error over
      // the bound is the type running out of room and never the grid slipping.
      CHECK(e <= (bounds[r] + half_ulp_rel16(in[i])) * 1.0001);
      if (fabsf(x) < nmin) {
        ++nsub;
        if (e > wS)
          wS = e;
        if (e > bounds[r])
          ++overS;
      } else {
        if (e > wN)
          wN = e;
        if (e > bounds[r])
          ++overN;
      }
    }
    printf("  f16      %-20s normals worst %.6f over %d, subnormals(%d) worst "
           "%.6f over %d, sum %016" PRIx64 "\n",
           recipes[r], wN, overN, nsub, wS, overS, sum);
    CHECK(overN == 0);
  }
}

#define BLK 65536

static void every_normal_float32(void) {
  static float in[BLK], out[BLK];
  static int32_t st[BLK];
  const float nmin = (float)quant_log_normal_min(QLOG_F32);

  for (int r = 0; r < NREC; ++r) {
    quant_log_params p;
    if (setup(recipes[r], QLOG_F32, &p) != 0)
      continue;
    double worst = 0.0;
    uint32_t at = 0;
    uint64_t counted = 0, over = 0, sum = 0;

    for (uint64_t base = 0; base < 0x100000000ull; base += BLK) {
      for (int i = 0; i < BLK; ++i) {
        const uint32_t b = (uint32_t)(base + (uint64_t)i);
        memcpy(&in[i], &b, sizeof(float));
      }
      CHECK(quant_log_encode(st, in, &p, QLOG_F32, BLK) == 0);
      CHECK(quant_log_decode(out, st, &p, QLOG_F32, BLK) == 0);
      for (int i = 0; i < BLK; ++i) {
        uint32_t rb;
        memcpy(&rb, &out[i], sizeof(rb));
        FOLD(sum, rb, base + (uint64_t)i);
        const float x = in[i];
        if (!isfinite(x))
          continue;
        if (x == 0.0f) {
          CHECK(out[i] == 0.0f);
          continue;
        }
        if (fabsf(x) < nmin)
          continue;
        ++counted;
        const double e = fabs((double)out[i] - (double)x) / fabs((double)x);
        if (e > worst) {
          worst = e;
          memcpy(&at, &x, sizeof(uint32_t));
        }
        if (e > bounds[r])
          ++over;
      }
    }
    printf("  f32      %-20s %llu normals, worst %.6e of %.6e, over %llu, "
           "sum %016" PRIx64 "\n",
           recipes[r], (unsigned long long)counted, worst, bounds[r],
           (unsigned long long)over, sum);
    CHECK(over == 0);
    // A grid using a fraction of what it was given would pass the line above and
    // still be wrong, with levels closer than asked and an index longer than
    // needed.
    CHECK(worst > 0.99 * bounds[r]);
  }
}

int main(void) {
  printf("every value of a type, one at a time\n");
  every_u8();
  every_u16();
  every_i16();
  every_half();
  every_normal_float32();
  if (failures != 0) {
    printf("\n%d failed\n", failures);
    return 1;
  }
  printf("\nall passed\n");
  return 0;
}
