#include "planar/encode_planar_kernel.h"
#include "planar_zigzag/decode_planar_zigzag_kernel.h"
#include "planar_zigzag/encode_planar_zigzag_kernel.h"

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

#define N (13 * 17)

static uint64_t source[N], residual[N], expected[N], encoded[N], decoded[N];

#define ONE_WIDTH(T, B, W)                                                     \
  do {                                                                         \
    CHECK(planar_encode(residual, source, (W), N, sizeof(T)) == 0);             \
    T *r = (T *)residual;                                                       \
    T *want = (T *)expected;                                                    \
    for (size_t i = 0; i < N; ++i) {                                           \
      T sign = (T)(0 - (r[i] >> ((B)-1)));                                     \
      want[i] = (T)((T)(r[i] << 1) ^ sign);                                    \
    }                                                                          \
    CHECK(planar_zigzag_encode(encoded, source, (W), N, sizeof(T)) == 0);       \
    CHECK(memcmp(encoded, expected, N * sizeof(T)) == 0);                       \
    CHECK(planar_zigzag_decode(decoded, encoded, (W), N, sizeof(T)) == 0);      \
    CHECK(memcmp(decoded, source, N * sizeof(T)) == 0);                         \
    memcpy(decoded, encoded, N * sizeof(T));                                   \
    CHECK(planar_zigzag_decode(decoded, decoded, (W), N, sizeof(T)) == 0);      \
    CHECK(memcmp(decoded, source, N * sizeof(T)) == 0);                         \
  } while (0)

#define ONE_TYPE(T, B)                                                         \
  do {                                                                         \
    uint64_t state = UINT64_C(0x9e3779b97f4a7c15);                             \
    T *s = (T *)source;                                                        \
    for (size_t i = 0; i < N; ++i) {                                           \
      state = state * UINT64_C(6364136223846793005) + 1;                       \
      s[i] = (T)(state ^ (state >> 23));                                       \
    }                                                                          \
    ONE_WIDTH(T, B, 1);                                                        \
    ONE_WIDTH(T, B, 13);                                                       \
    ONE_WIDTH(T, B, 17);                                                       \
    ONE_WIDTH(T, B, N);                                                        \
  } while (0)

int main(void) {
  printf("test_planar_zigzag\n");
  ONE_TYPE(uint8_t, 8);
  ONE_TYPE(uint16_t, 16);
  ONE_TYPE(uint32_t, 32);
  ONE_TYPE(uint64_t, 64);

  CHECK(planar_zigzag_encode(encoded, source, 0, N, 2) != 0);
  CHECK(planar_zigzag_decode(decoded, encoded, 12, N, 2) != 0);
  CHECK(planar_zigzag_decode(decoded, encoded, 13, N, 3) != 0);

  if (failures) {
    printf("test_planar_zigzag: %d failures\n", failures);
    return 1;
  }
  printf("test_planar_zigzag: ok\n");
  return 0;
}

#undef ONE_TYPE
#undef ONE_WIDTH
#undef N
