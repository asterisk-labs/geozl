// IEEE half conversions, for platforms without _Float16.

#ifndef GEOZL_CODECS_QUANT_LINEAR_HALF_H
#define GEOZL_CODECS_QUANT_LINEAR_HALF_H

#include <stdint.h>
#include <string.h>

static inline float quant_linear_half_to_float(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t man = h & 0x3FFu;
  uint32_t bits;
  if (exp == 0) {
    if (man == 0) {
      bits = sign;
    } else {
      exp = 127 - 15 + 1;
      while ((man & 0x400u) == 0) {
        man <<= 1;
        exp--;
      }
      man &= 0x3FFu;
      bits = sign | (exp << 23) | (man << 13);
    }
  } else if (exp == 0x1F) {
    bits = sign | 0x7F800000u | (man << 13);
  } else {
    bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
  }
  float f;
  memcpy(&f, &bits, sizeof(f));
  return f;
}

// Ties away from zero, not to even. Output path only. Kept as the reference the
// two below are checked against, and off every hot path.
static inline uint16_t quant_linear_float_to_half(float f) {
  uint32_t x;
  memcpy(&x, &f, sizeof(x));
  uint32_t sign = (x >> 16) & 0x8000u;
  int32_t exp = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
  uint32_t man = x & 0x7FFFFFu;
  if (((x >> 23) & 0xFFu) == 0xFFu)
    return (uint16_t)(sign | 0x7C00u | (man ? 0x200u : 0));
  if (exp >= 0x1F)
    return (uint16_t)(sign | 0x7C00u);
  if (exp <= 0) {
    if (exp < -10)
      return (uint16_t)sign;
    man |= 0x800000u;
    uint32_t shift = (uint32_t)(14 - exp);
    uint32_t h = man >> shift;
    if ((man >> (shift - 1)) & 1u)
      h += 1;
    return (uint16_t)(sign | h);
  }
  uint16_t h = (uint16_t)(sign | ((uint32_t)exp << 10) | (man >> 13));
  if (man & 0x1000u)
    h = (uint16_t)(h + 1);
  return h;
}

// For a normal float, (ax - QL_HALF_REBASE) >> 13 is the rebased exponent and
// the truncated mantissa in one subtract and one shift.
#define QL_HALF_REBASE (((uint32_t)(127 - 15)) << 23)

// The int16 stream of the STORE=VALUES path. Exact in float, so no nan, no
// infinity, no subnormal and no overflow. Only the rebase is left.
static inline uint16_t ql_i16_to_half(int32_t v) {
  const float f = (float)v;
  uint32_t x;
  memcpy(&x, &f, sizeof(x));
  const uint32_t sign = (x >> 16) & 0x8000u;
  const uint32_t ax = x & 0x7FFFFFFFu;
  uint32_t h = (ax - QL_HALF_REBASE) >> 13;
  h += (ax >> 12) & 1u; // the carry runs into the exponent, which is correct
  return (uint16_t)(sign | (ax == 0 ? 0u : h));
}

// Same bits as quant_linear_float_to_half, without branches. One branch in here
// stops the vectoriser on the loop that calls it, which is the whole point.
static inline uint16_t ql_f32_to_half(float f) {
  uint32_t x;
  memcpy(&x, &f, sizeof(x));
  const uint32_t sign = (x >> 16) & 0x8000u;
  const uint32_t ax = x & 0x7FFFFFFFu;
  const uint32_t e = ax >> 23;

  const uint32_t mSub = (uint32_t)0 - (e <= 112u);
  const uint32_t mBig = (uint32_t)0 - (e >= 143u);
  const uint32_t mNan =
      ((uint32_t)0 - (e == 255u)) & ((uint32_t)0 - ((ax & 0x7FFFFFu) != 0));

  uint32_t hn = (ax - QL_HALF_REBASE) >> 13;
  hn += (ax >> 12) & 1u;

  // Masked, or the cast sees a value past uint32 where this half of the select
  // was not chosen. Undefined, and what float-cast-overflow catches.
  const uint32_t axs = ax & mSub;
  float af;
  memcpy(&af, &axs, sizeof(af));
  // A subnormal half is k * 2^-24 for whole k, so the scaling is exact. Adding
  // the half before truncating rounds on its own at 0x32ffffff.
  const float v = af * 16777216.0f;
  uint32_t hs = (uint32_t)v;
  hs += (v - (float)hs) >= 0.5f;

  uint32_t h = (hs & mSub) | (hn & ~mSub);
  h = (h & ~mBig) | ((0x7C00u | (0x200u & mNan)) & mBig);
  return (uint16_t)(sign | h);
}

#endif // GEOZL_CODECS_QUANT_LINEAR_HALF_H
