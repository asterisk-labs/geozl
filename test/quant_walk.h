#ifndef GEOZL_TEST_QUANT_WALK_H
#define GEOZL_TEST_QUANT_WALK_H

#include "geozl/dtype.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Every reconstruction folded into one number. Addition, so the order it is fed
// in cannot change the result. Two builds that agree here came back with the
// same bits on every value, which matching worst cases alone does not say.
#define GEOZL_FOLD(sum, bits, pos)                                             \
  ((sum) += (uint64_t)(bits) * 1099511628211ull + (uint64_t)(pos))

#define GEOZL_WALK_BLK 65536

// f32 has 2^32 values and walking them takes minutes, so it goes on a stride
// unless asked. 4093 is prime, so it lands on every residue class and never
// lines up with an exponent boundary.
static inline uint64_t geozl_walk_step(void) {
  const char *e = getenv("GEOZL_EXHAUSTIVE");
  return (e != NULL && e[0] != '\0' && e[0] != '0') ? 1 : 4093;
}

// Smallest domain first, so a break shows up on the cheap types before the
// expensive one. f16 is here as a bit pattern like the rest; what it means is
// the codec's business.
static const struct {
  int dtype;
  const char *name;
  uint64_t domain;
  size_t width;
} geozl_walk_types[] = {
    {GEOZL_DT_U8, "u8", 1ull << 8, 1},
    {GEOZL_DT_U16, "u16", 1ull << 16, 2},
    {GEOZL_DT_I16, "i16", 1ull << 16, 2},
    {GEOZL_DT_F16, "f16", 1ull << 16, 2},
    {GEOZL_DT_F32, "f32", 1ull << 32, 4},
};
#define GEOZL_WALK_NTYPES                                                      \
  (sizeof(geozl_walk_types) / sizeof(geozl_walk_types[0]))

// The block starting at base, strided. Returns how many values it wrote, which
// is short only on the last block of a domain.
static inline size_t geozl_walk_fill(void *dst, int dtype, uint64_t base,
                                     uint64_t domain, uint64_t step) {
  size_t n = 0;
  for (; n < GEOZL_WALK_BLK; ++n) {
    const uint64_t v = base + (uint64_t)n * step;
    if (v >= domain)
      break;
    switch (dtype) {
    case GEOZL_DT_U8:
      ((uint8_t *)dst)[n] = (uint8_t)v;
      break;
    case GEOZL_DT_I16:
      ((int16_t *)dst)[n] = (int16_t)(uint16_t)v;
      break;
    case GEOZL_DT_F32: {
      const uint32_t b = (uint32_t)v;
      memcpy((char *)dst + n * 4, &b, 4);
      break;
    }
    default: // u16 and f16 are both a 16 bit pattern
      ((uint16_t *)dst)[n] = (uint16_t)v;
      break;
    }
  }
  return n;
}

// The half decode, kept here rather than borrowed from a codec so the walk does
// not measure the codec's own conversion against itself.
static inline double geozl_walk_half(uint16_t h) {
  const int e = (h >> 10) & 0x1F;
  const int m = h & 0x3FF;
  const double s = (h & 0x8000) ? -1.0 : 1.0;
  if (e == 0) // subnormal, m * 2^-24
    return s * (double)m * 5.9604644775390625e-8;
  if (e == 31)
    return m ? (0.0 / 0.0) : s * (1.0 / 0.0);
  // 2^(e-15) * (1 + m/1024), which is (1024 + m) * 2^(e-25)
  return s * (double)(m | 0x400) * 2.98023223876953125e-8 * (double)(1u << e);
}

static inline double geozl_walk_get(const void *p, int dtype, size_t i) {
  switch (dtype) {
  case GEOZL_DT_U8:
    return ((const uint8_t *)p)[i];
  case GEOZL_DT_U16:
    return ((const uint16_t *)p)[i];
  case GEOZL_DT_I16:
    return ((const int16_t *)p)[i];
  case GEOZL_DT_F16:
    return geozl_walk_half(((const uint16_t *)p)[i]);
  default:
    return ((const float *)p)[i];
  }
}

// The raw bits of a reconstruction, which is what the checksum folds.
static inline uint64_t geozl_walk_bits(const void *p, int dtype, size_t i) {
  switch (dtype) {
  case GEOZL_DT_U8:
    return ((const uint8_t *)p)[i];
  case GEOZL_DT_F32: {
    uint32_t b;
    memcpy(&b, (const char *)p + i * 4, 4);
    return b;
  }
  default:
    return ((const uint16_t *)p)[i];
  }
}

#endif // GEOZL_TEST_QUANT_WALK_H