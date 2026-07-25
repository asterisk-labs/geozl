// Byte order of the codec header helpers.
//
// The point of these checks is the byte positions (regression on a big-endian target without owning one).

#include "common/endian.h"

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

static void test_scalar_layout(void) {
  uint8_t b[16];

  geozl_st_le16(b, 0xBEEFu);
  CHECK(b[0] == 0xEF && b[1] == 0xBE);
  CHECK(geozl_ld_le16(b) == 0xBEEFu);

  geozl_st_le32(b, 0x11223344u);
  CHECK(b[0] == 0x44 && b[1] == 0x33 && b[2] == 0x22 && b[3] == 0x11);
  CHECK(geozl_ld_le32(b) == 0x11223344u);

  geozl_st_le64(b, 0x0123456789ABCDEFull);
  CHECK(b[0] == 0xEF && b[7] == 0x01);
  CHECK(geozl_ld_le64(b) == 0x0123456789ABCDEFull);
}

// wp_static carries four int16 coefficients and the planar default is
// {1, -1, 0, 0}, so the negative path is the common one, not the edge case.
static void test_signed_coeffs(void) {
  static const int16_t values[] = { 0, 1, -1, 32767, -32768, -2 };
  uint8_t b[2];

  for (size_t i = 0; i < sizeof(values) / sizeof(values[0]); ++i) {
    geozl_st_le_i16(b, values[i]);
    CHECK(geozl_ld_le_i16(b) == values[i]);
  }

  geozl_st_le_i16(b, -2);
  CHECK(b[0] == 0xFE && b[1] == 0xFF);
}

// intmult stores its base and binoffset its bin lowers at the sample width,
// so the variable-width pair has to stay exact and must not write past n.
static void test_variable_width(void) {
  uint8_t b[16];

  for (size_t n = 1; n <= 8; n <<= 1) {
    const uint64_t mask = (n == 8) ? ~(uint64_t)0 : (((uint64_t)1 << (8 * n)) - 1);
    const uint64_t v = 0x8877665544332211ull & mask;

    memset(b, 0xAA, sizeof(b));
    geozl_st_le(b, v, n);
    CHECK(geozl_ld_le(b, n) == v);
    CHECK(b[0] == (uint8_t)v);
    CHECK(b[n] == 0xAA);
  }
}

// floatmult and quant_linear put an IEEE-754 double in the header. Compared
// bitwise so a signed zero or a subnormal is not waved through by ==.
static void test_double_bits(void) {
  static const double values[] = { 0.0, -0.0, 1.0, -0.75, 0.1,
                                   1e-300, 1e300, 5e-324 };
  uint8_t b[8];

  geozl_st_le_f64(b, -0.75);
  CHECK(b[7] == 0xBF && b[6] == 0xE8);

  for (size_t i = 0; i < sizeof(values) / sizeof(values[0]); ++i) {
    geozl_st_le_f64(b, values[i]);
    const double got = geozl_ld_le_f64(b);
    CHECK(memcmp(&got, &values[i], sizeof(got)) == 0);
  }
}

int main(void) {
  test_scalar_layout();
  test_signed_coeffs();
  test_variable_width();
  test_double_bits();

  if (failures) {
    printf("test_endian: %d failed\n", failures);
    return 1;
  }
  printf("test_endian: ok\n");
  return 0;
}
