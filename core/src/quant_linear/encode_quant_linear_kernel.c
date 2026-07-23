#include "encode_quant_linear_kernel.h"
#include "quant_linear_dtype.h" // ql_dtype

#include <math.h>
#include <stdint.h>
#include <string.h>

// q = round(x / scale). Integer input uses exact integer arithmetic so big 64
// bit values do not lose precision through a double. Float input divides in
// double and rounds to the nearest integer, then casts to a signed integer of
// the same width, saturating so a pathological value cannot wrap. scale is even
// for integers and a real step for floats, so the error is bounded by scale/2 =
// max_error per element.

// Unsigned integer, q = (x + step/2) / step. The index shares the sample width,
// so a step below 1 cannot fit it and gains nothing on integers, clamp it to 1.
#define QL_ENC_U(T)                                                            \
  do {                                                                         \
    const T *s = (const T *)src;                                               \
    T *d = (T *)dst;                                                           \
    const uint64_t isc = (uint64_t)(scale < 1.0 ? 1.0 : scale);                \
    const uint64_t half = isc >> 1;                                            \
    for (size_t i = 0; i < nbElts; ++i)                                        \
      d[i] = (T)(((unsigned __int128)s[i] + half) / isc);                      \
  } while (0)

// Signed integer, round half away from zero on the magnitude, keep the sign.
#define QL_ENC_I(T)                                                            \
  do {                                                                         \
    const T *s = (const T *)src;                                               \
    T *d = (T *)dst;                                                           \
    const int64_t isc = (int64_t)(scale < 1.0 ? 1.0 : scale);                  \
    const int64_t half = isc >> 1;                                             \
    for (size_t i = 0; i < nbElts; ++i) {                                      \
      const __int128 v = (__int128)s[i];                                       \
      const __int128 m = v < 0 ? -v : v;                                       \
      const __int128 q = (m + half) / isc;                                     \
      d[i] = (T)(int64_t)(v < 0 ? -q : q);                                     \
    }                                                                          \
  } while (0)

// Store the reconstruction instead of the index, so the decoder is a copy. The
// result is exactly what QL_DEC_U would have produced from the index.
#define QL_ENC_UV(T)                                                           \
  do {                                                                         \
    const T *s = (const T *)src;                                               \
    T *d = (T *)dst;                                                           \
    const uint64_t isc = (uint64_t)(scale < 1.0 ? 1.0 : scale);                \
    const uint64_t half = isc >> 1;                                            \
    const uint64_t cap = (uint64_t)(T)(~(T)0);                                 \
    for (size_t i = 0; i < nbElts; ++i) {                                      \
      const uint64_t q = (uint64_t)(((unsigned __int128)s[i] + half) / isc);   \
      const unsigned __int128 r = (unsigned __int128)q * isc;                  \
      d[i] = (T)(r < cap ? (uint64_t)r : cap);                                 \
    }                                                                          \
  } while (0)

// Signed counterpart, matching QL_DEC_I including the clamp at both ends.
#define QL_ENC_IV(T, LO, HI)                                                   \
  do {                                                                         \
    const T *s = (const T *)src;                                               \
    T *d = (T *)dst;                                                           \
    const int64_t isc = (int64_t)(scale < 1.0 ? 1.0 : scale);                  \
    const int64_t half = isc >> 1;                                             \
    for (size_t i = 0; i < nbElts; ++i) {                                      \
      const __int128 v = (__int128)s[i];                                       \
      const __int128 m = v < 0 ? -v : v;                                       \
      const __int128 q = v < 0 ? -((m + half) / isc) : (m + half) / isc;       \
      __int128 r = q * isc;                                                    \
      if (r < (LO))                                                            \
        r = (LO);                                                              \
      if (r > (HI))                                                            \
        r = (HI);                                                              \
      d[i] = (T)(int64_t)r;                                                    \
    }                                                                          \
  } while (0)

// Float of value width WT to a signed integer index of the same width IT.
#define QL_ENC_F(WT, IT, MINV, MAXV)                                           \
  do {                                                                         \
    const WT *s = (const WT *)src;                                             \
    IT *d = (IT *)dst;                                                         \
    for (size_t i = 0; i < nbElts; ++i) {                                      \
      double q = nearbyint((double)s[i] / scale);                              \
      if (q < (double)(MINV))                                                  \
        q = (double)(MINV);                                                    \
      if (q > (double)(MAXV))                                                  \
        q = (double)(MAXV);                                                    \
      d[i] = (IT)q;                                                            \
    }                                                                          \
  } while (0)

// IEEE half to float, for platforms without native _Float16.
static float ql_half_to_float(uint16_t h) {
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

void quant_linear_encode(void *dst, const void *src, double scale, int dtype,
                         size_t nbElts) {
  if (dtype < QL_U8 || dtype > QL_F64)
    return;
  // A negative scale means store the reconstruction, not the index, so the
  // decoder only copies. Only integers can do that, their reconstruction has
  // the same type as the input. The step itself is the magnitude.
  const int store_values = scale < 0.0 && dtype <= QL_I64;
  if (scale < 0.0)
    scale = -scale;
  // scale 0 is exact. scale 1 is exact too on integers, where the index is
  // the value itself, so both skip the loop.
  if (scale == 0.0 || (scale == 1.0 && dtype <= QL_I64)) {
    size_t w[] = {1, 2, 4, 8, 1, 2, 4, 8, 2, 4, 8};
    memcpy(dst, src, nbElts * w[dtype]);
    return;
  }
  if (store_values) {
    switch ((ql_dtype)dtype) {
    case QL_U8:
      QL_ENC_UV(uint8_t);
      break;
    case QL_U16:
      QL_ENC_UV(uint16_t);
      break;
    case QL_U32:
      QL_ENC_UV(uint32_t);
      break;
    case QL_U64:
      QL_ENC_UV(uint64_t);
      break;
    case QL_I8:
      QL_ENC_IV(int8_t, INT8_MIN, INT8_MAX);
      break;
    case QL_I16:
      QL_ENC_IV(int16_t, INT16_MIN, INT16_MAX);
      break;
    case QL_I32:
      QL_ENC_IV(int32_t, INT32_MIN, INT32_MAX);
      break;
    case QL_I64:
      QL_ENC_IV(int64_t, INT64_MIN, INT64_MAX);
      break;
    default:
      break;
    }
    return;
  }
  switch ((ql_dtype)dtype) {
  case QL_U8:
    QL_ENC_U(uint8_t);
    break;
  case QL_U16:
    QL_ENC_U(uint16_t);
    break;
  case QL_U32:
    QL_ENC_U(uint32_t);
    break;
  case QL_U64:
    QL_ENC_U(uint64_t);
    break;
  case QL_I8:
    QL_ENC_I(int8_t);
    break;
  case QL_I16:
    QL_ENC_I(int16_t);
    break;
  case QL_I32:
    QL_ENC_I(int32_t);
    break;
  case QL_I64:
    QL_ENC_I(int64_t);
    break;
  case QL_F16: {
    const uint16_t *s = (const uint16_t *)src;
    int16_t *d = (int16_t *)dst;
    for (size_t i = 0; i < nbElts; ++i) {
      double q = nearbyint((double)ql_half_to_float(s[i]) / scale);
      if (q < -32768.0)
        q = -32768.0;
      if (q > 32767.0)
        q = 32767.0;
      d[i] = (int16_t)q;
    }
    break;
  }
  case QL_F32:
    QL_ENC_F(float, int32_t, INT32_MIN, INT32_MAX);
    break;
  case QL_F64:
    QL_ENC_F(double, int64_t, INT64_MIN, INT64_MAX);
    break;
  }
}