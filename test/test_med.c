#include "med/decode_med_kernel.h"
#include "med/encode_med_kernel.h"

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

#define MAXN 1024

static uint64_t source[MAXN], residual[MAXN], expected[MAXN], decoded[MAXN];

#define FILL(T, buf, n)                                                        \
  do {                                                                         \
    T *p = (T *)(buf);                                                         \
    for (size_t i = 0; i < (n); ++i) {                                         \
      state = state * UINT64_C(6364136223846793005) + 1;                       \
      p[i] = (T)(state ^ (state >> 23));                                       \
    }                                                                          \
  } while (0)

// The spec's own spelling, ITU-T T.87 A.4.2, one row after the other. The
// kernel reorders the traversal and writes the median with masks, so this
// stays an independent reading of the same rule.
#define REF_DEC(T, W, H)                                                       \
  do {                                                                         \
    T *ref = (T *)expected;                                                    \
    const T *res = (const T *)residual;                                        \
    ref[0] = res[0];                                                           \
    for (size_t c = 1; c < (W); ++c)                                           \
      ref[c] = (T)(res[c] + ref[c - 1]);                                       \
    for (size_t r = 1; r < (H); ++r)                                           \
      for (size_t c = 0; c < (W); ++c) {                                       \
        const size_t i = r * (W) + c;                                          \
        const T Wv = (c > 0) ? ref[i - 1] : 0;                                 \
        const T Nv = ref[i - (W)];                                             \
        const T NWv = (c > 0) ? ref[i - (W)-1] : 0;                            \
        const T mn = Wv < Nv ? Wv : Nv;                                        \
        const T mx = Wv < Nv ? Nv : Wv;                                        \
        const T P = (NWv >= mx) ? mn : (NWv <= mn) ? mx : (T)(Wv + Nv - NWv);  \
        ref[i] = (T)(res[i] + P);                                              \
      }                                                                        \
  } while (0)

// Decode is checked on arbitrary residuals, not only the ones an encoder would
// produce, so the wavefront has to agree with the reference where the
// arithmetic wraps too.
#define ONE_SHAPE(T, W, H)                                                     \
  do {                                                                         \
    const size_t n = (size_t)(W) * (H);                                        \
    FILL(T, residual, n);                                                      \
    REF_DEC(T, W, H);                                                          \
    CHECK(med_decode(decoded, residual, (W), n, sizeof(T)) == 0);              \
    CHECK(memcmp(decoded, expected, n * sizeof(T)) == 0);                      \
    memcpy(decoded, residual, n * sizeof(T));                                  \
    CHECK(med_decode(decoded, decoded, (W), n, sizeof(T)) == 0);               \
    CHECK(memcmp(decoded, expected, n * sizeof(T)) == 0);                      \
    FILL(T, source, n);                                                        \
    CHECK(med_encode(residual, source, (W), n, sizeof(T)) == 0);               \
    CHECK(med_decode(decoded, residual, (W), n, sizeof(T)) == 0);              \
    CHECK(memcmp(decoded, source, n * sizeof(T)) == 0);                        \
  } while (0)

// The widths straddle the guard at twice the block height, and the row counts
// leave the serial tail every remainder from none to three.
#define ONE_TYPE(T)                                                            \
  do {                                                                         \
    uint64_t state = UINT64_C(0x9e3779b97f4a7c15);                             \
    for (size_t h = 1; h <= 13; ++h) {                                         \
      ONE_SHAPE(T, 1, h);                                                      \
      ONE_SHAPE(T, 2, h);                                                      \
      ONE_SHAPE(T, 4, h);                                                      \
      ONE_SHAPE(T, 7, h);                                                      \
      ONE_SHAPE(T, 8, h);                                                      \
      ONE_SHAPE(T, 9, h);                                                      \
      ONE_SHAPE(T, 11, h);                                                     \
      ONE_SHAPE(T, 16, h);                                                     \
    }                                                                          \
    ONE_SHAPE(T, 64, 15);                                                      \
    ONE_SHAPE(T, 127, 8);                                                      \
    ONE_SHAPE(T, 1024, 1);                                                     \
  } while (0)

int main(void) {
  printf("test_med\n");
  ONE_TYPE(uint8_t);
  ONE_TYPE(uint16_t);
  ONE_TYPE(uint32_t);
  ONE_TYPE(uint64_t);

  CHECK(med_decode(decoded, residual, 0, 64, 2) != 0);
  CHECK(med_decode(decoded, residual, 64, 0, 2) != 0);
  CHECK(med_decode(decoded, residual, 5, 64, 2) != 0);
  CHECK(med_decode(decoded, residual, 8, 64, 3) != 0);

  if (failures) {
    printf("test_med: %d failures\n", failures);
    return 1;
  }
  printf("test_med: ok\n");
  return 0;
}
