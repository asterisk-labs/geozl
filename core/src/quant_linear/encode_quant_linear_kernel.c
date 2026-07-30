#include "encode_quant_linear_kernel.h"

#include "quant_linear_dtype.h"
#include "quant_linear_half.h"

#include <math.h>
#include <stdint.h>

static uint64_t ql_step_u64(double step) {
  const double s = floor(step);
  return s < 1.0 ? 1u : (s > 18446744073709549568.0 ? UINT64_MAX : (uint64_t)s);
}

static double ql_fit(double q, double lo, double hi) {
  if (q != q)
    return 0.0;
  return q < lo ? lo : (q > hi ? hi : q);
}

// The quotient and the product are taken in 128 bits because a u64 value plus half
// a step overflows 64 on the way.
#define QL_ENC_UV(T)                                                           \
  do {                                                                         \
    const T *s = (const T *)src;                                               \
    T *d = (T *)dst;                                                           \
    const uint64_t isc = ql_step_u64(step);                                    \
    const uint64_t half = isc >> 1;                                            \
    const uint64_t cap = (uint64_t)(T)(~(T)0);                                 \
    for (size_t i = 0; i < nbElts; ++i) {                                      \
      const unsigned __int128 r =                                              \
          ((unsigned __int128)s[i] + half) / isc * isc;                        \
      d[i] = (T)(r < cap ? (uint64_t)r : cap);                                 \
    }                                                                          \
  } while (0)

// Rounding is done on the magnitude and the sign put back, so the grid stays
// symmetric about zero.
#define QL_ENC_IV(T, LO, HI)                                                   \
  do {                                                                         \
    const T *s = (const T *)src;                                               \
    T *d = (T *)dst;                                                           \
    const uint64_t isc = ql_step_u64(step);                                    \
    const uint64_t half = isc >> 1;                                            \
    for (size_t i = 0; i < nbElts; ++i) {                                      \
      const __int128 v = (__int128)s[i];                                       \
      const unsigned __int128 m =                                              \
          v < 0 ? (unsigned __int128)(-v) : (unsigned __int128)v;              \
      __int128 r = (__int128)((m + half) / isc * isc);                         \
      if (v < 0)                                                              \
        r = -r;                                                                \
      d[i] = r < (__int128)(LO) ? (LO) : (r > (__int128)(HI) ? (HI) : (T)r);   \
    }                                                                          \
  } while (0)

// With STORE=VALUES the step is whole, so q*step is too and the same integer
// stream holds it.
#define QL_ENC_F(WT, IT, RD)                                                   \
  do {                                                                         \
    const WT *s = (const WT *)src;                                             \
    IT *d = (IT *)dst;                                                         \
    for (size_t i = 0; i < nbElts; ++i) {                                      \
      const double q = nearbyint((double)(RD) / step);                          \
      d[i] = (IT)(int64_t)ql_fit(values ? q * step : q, -smax - 1.0, smax);    \
    }                                                                          \
  } while (0)

int quant_linear_encode(void *dst, const void *src,
                        const quant_linear_params *p, int dtype,
                        size_t nbElts) {
  if (!QL_DTYPE_OK(dtype) || !(p->step > 0.0))
    return 1;
  const double step = p->step;
  const double smax = quant_linear_stream_max(dtype);
  const int values = (p->flags & QUANT_LINEAR_FLAG_STORE_VALUES) != 0;
  if (dtype > QL_LAST_INT && values && floor(step) != step)
    return 1;

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
  case QL_F16:
    QL_ENC_F(uint16_t, int16_t, quant_linear_half_to_float(s[i]));
    break;
  case QL_F32:
    QL_ENC_F(float, int32_t, s[i]);
    break;
  case QL_F64:
    QL_ENC_F(double, int64_t, s[i]);
    break;
  }
  return 0;
}

#define QL_SCAN(RD)                                                            \
  do {                                                                         \
    for (size_t i = 0; i < nbElts; ++i) {                                      \
      const double v = (double)(RD);                                           \
      if (!isfinite(v))                                                        \
        continue;                                                              \
      if (v < 0.0)                                                             \
        neg = 1;                                                               \
      const double a = fabs(v);                                                \
      if (a > hi)                                                              \
        hi = a;                                                                \
    }                                                                          \
  } while (0)

int quant_linear_scan(const void *src, int dtype, size_t nbElts, double *maxAbs,
                      int *anyNegative) {
  double hi = 0.0;
  int neg = 0;
  if (!QL_DTYPE_OK(dtype))
    return 1;

  switch ((ql_dtype)dtype) {
  case QL_U8:
    QL_SCAN(((const uint8_t *)src)[i]);
    break;
  case QL_U16:
    QL_SCAN(((const uint16_t *)src)[i]);
    break;
  case QL_U32:
    QL_SCAN(((const uint32_t *)src)[i]);
    break;
  case QL_U64:
    QL_SCAN(((const uint64_t *)src)[i]);
    break;
  case QL_I8:
    QL_SCAN(((const int8_t *)src)[i]);
    break;
  case QL_I16:
    QL_SCAN(((const int16_t *)src)[i]);
    break;
  case QL_I32:
    QL_SCAN(((const int32_t *)src)[i]);
    break;
  case QL_I64:
    QL_SCAN(((const int64_t *)src)[i]);
    break;
  case QL_F16:
    QL_SCAN(quant_linear_half_to_float(((const uint16_t *)src)[i]));
    break;
  case QL_F32:
    QL_SCAN(((const float *)src)[i]);
    break;
  case QL_F64:
    QL_SCAN(((const double *)src)[i]);
    break;
  }

  if (maxAbs != NULL)
    *maxAbs = hi;
  if (anyNegative != NULL)
    *anyNegative = neg;
  return hi > 0.0 ? 0 : 1;
}
