#include "encode_quant_kernel.h"
#include "quant_half.h" // quant_half_to_float

#include <stdint.h>
#include <string.h>

// The linear curve stays on exact integer arithmetic so a 64 bit sample does
// not lose precision through a double. The warped curves go through double.

#define Q_ENC_U(T)                                                             \
  do {                                                                         \
    const T *s = (const T *)src;                                               \
    T *d = (T *)dst;                                                           \
    const uint64_t isc = quant_step_u64(step);                                 \
    const uint64_t half = isc >> 1;                                            \
    for (size_t i = 0; i < nbElts; ++i)                                        \
      d[i] = (T)(((unsigned __int128)s[i] + half) / isc);                      \
  } while (0)

#define Q_ENC_I(T)                                                             \
  do {                                                                         \
    const T *s = (const T *)src;                                               \
    T *d = (T *)dst;                                                           \
    const int64_t isc = quant_step_i64(step);                                  \
    const int64_t half = isc >> 1;                                             \
    for (size_t i = 0; i < nbElts; ++i) {                                      \
      const __int128 v = (__int128)s[i];                                       \
      const __int128 m = v < 0 ? -v : v;                                       \
      const __int128 q = (m + half) / isc;                                     \
      d[i] = (T)(int64_t)(v < 0 ? -q : q);                                     \
    }                                                                          \
  } while (0)

// Store the reconstruction instead of the index, so the decoder only copies.
#define Q_ENC_UV(T)                                                            \
  do {                                                                         \
    const T *s = (const T *)src;                                               \
    T *d = (T *)dst;                                                           \
    const uint64_t isc = quant_step_u64(step);                                 \
    const uint64_t half = isc >> 1;                                            \
    const uint64_t cap = (uint64_t)(T)(~(T)0);                                 \
    for (size_t i = 0; i < nbElts; ++i) {                                      \
      const uint64_t q = (uint64_t)(((unsigned __int128)s[i] + half) / isc);   \
      const unsigned __int128 r = (unsigned __int128)q * isc;                  \
      d[i] = (T)(r < cap ? (uint64_t)r : cap);                                 \
    }                                                                          \
  } while (0)

#define Q_ENC_IV(T, LO, HI)                                                    \
  do {                                                                         \
    const T *s = (const T *)src;                                               \
    T *d = (T *)dst;                                                           \
    const int64_t isc = quant_step_i64(step);                                  \
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

#define Q_ENC_LIN_F(WT, IT)                                                    \
  do {                                                                         \
    const WT *s = (const WT *)src;                                             \
    IT *d = (IT *)dst;                                                         \
    for (size_t i = 0; i < nbElts; ++i)                                        \
      d[i] = (IT)quant_index_fit(nearbyint((double)s[i] / step), ilo, ihi);    \
  } while (0)

// Pick the index whose reconstruction, at the output width, is nearest the
// sample. quant_fwd lands on it or one short, and the curves are monotone, so
// only the neighbour on the side of the residual can be closer. Checking it
// keeps the bound off the accuracy of log and exp.
#define Q_ENC_WARP(RD, RT, IT)                                                 \
  do {                                                                         \
    IT *d = (IT *)dst;                                                         \
    for (size_t i = 0; i < nbElts; ++i) {                                      \
      const double x = (double)(RD);                                           \
      double q = quant_fwd(x, p);                                              \
      const double r0 = RT(quant_inv(q, p), vlo, vhi);                         \
      const double alt = q + (x > r0 ? 1.0 : -1.0);                            \
      if (fabs(x - RT(quant_inv(alt, p), vlo, vhi)) < fabs(x - r0))            \
        q = alt;                                                               \
      d[i] = (IT)quant_index_fit(q, ilo, ihi);                                 \
    }                                                                          \
  } while (0)

int quant_encode(void *dst, const void *src, const quant_params *p, int dtype,
                 size_t nbElts) {
  if (dtype < Q_U8 || dtype > Q_F64)
    return 1;
  if (p->curve > QUANT_CURVE_LOG)
    return 1;

  const double step = p->step;
  // A step of zero is exact, and so is a step of one on integers under the
  // linear curve, where the index and the value are the same number.
  if (step == 0.0 ||
      (p->curve == QUANT_CURVE_LINEAR && step == 1.0 && dtype <= Q_LAST_INT)) {
    static const size_t w[] = {1, 2, 4, 8, 1, 2, 4, 8, 2, 4, 8};
    memcpy(dst, src, nbElts * w[dtype]);
    return 0;
  }

  if (p->curve != QUANT_CURVE_LINEAR) {
    const double vlo = quant_value_lo(dtype), vhi = quant_value_hi(dtype);
    const double ilo = quant_index_lo(dtype), ihi = quant_index_hi(dtype);
    switch ((quant_dtype)dtype) {
    case Q_U8:
      Q_ENC_WARP(((const uint8_t *)src)[i], QUANT_RT_INT, uint8_t);
      break;
    case Q_U16:
      Q_ENC_WARP(((const uint16_t *)src)[i], QUANT_RT_INT, uint16_t);
      break;
    case Q_U32:
      Q_ENC_WARP(((const uint32_t *)src)[i], QUANT_RT_INT, uint32_t);
      break;
    case Q_U64:
      Q_ENC_WARP(((const uint64_t *)src)[i], QUANT_RT_INT, uint64_t);
      break;
    case Q_I8:
      Q_ENC_WARP(((const int8_t *)src)[i], QUANT_RT_INT, int8_t);
      break;
    case Q_I16:
      Q_ENC_WARP(((const int16_t *)src)[i], QUANT_RT_INT, int16_t);
      break;
    case Q_I32:
      Q_ENC_WARP(((const int32_t *)src)[i], QUANT_RT_INT, int32_t);
      break;
    case Q_I64:
      Q_ENC_WARP(((const int64_t *)src)[i], QUANT_RT_INT, int64_t);
      break;
    case Q_F16:
      Q_ENC_WARP(quant_half_to_float(((const uint16_t *)src)[i]), QUANT_RT_F16,
                 int16_t);
      break;
    case Q_F32:
      Q_ENC_WARP(((const float *)src)[i], QUANT_RT_F32, int32_t);
      break;
    case Q_F64:
      Q_ENC_WARP(((const double *)src)[i], QUANT_RT_F64, int64_t);
      break;
    }
    return 0;
  }

  const int store_values = (p->flags & QUANT_FLAG_STORE_VALUES) != 0;
  if (store_values) {
    if (dtype > Q_LAST_INT)
      return 1;
    switch ((quant_dtype)dtype) {
    case Q_U8:
      Q_ENC_UV(uint8_t);
      break;
    case Q_U16:
      Q_ENC_UV(uint16_t);
      break;
    case Q_U32:
      Q_ENC_UV(uint32_t);
      break;
    case Q_U64:
      Q_ENC_UV(uint64_t);
      break;
    case Q_I8:
      Q_ENC_IV(int8_t, INT8_MIN, INT8_MAX);
      break;
    case Q_I16:
      Q_ENC_IV(int16_t, INT16_MIN, INT16_MAX);
      break;
    case Q_I32:
      Q_ENC_IV(int32_t, INT32_MIN, INT32_MAX);
      break;
    case Q_I64:
      Q_ENC_IV(int64_t, INT64_MIN, INT64_MAX);
      break;
    default:
      break;
    }
    return 0;
  }

  const double ilo = quant_index_lo(dtype), ihi = quant_index_hi(dtype);
  switch ((quant_dtype)dtype) {
  case Q_U8:
    Q_ENC_U(uint8_t);
    break;
  case Q_U16:
    Q_ENC_U(uint16_t);
    break;
  case Q_U32:
    Q_ENC_U(uint32_t);
    break;
  case Q_U64:
    Q_ENC_U(uint64_t);
    break;
  case Q_I8:
    Q_ENC_I(int8_t);
    break;
  case Q_I16:
    Q_ENC_I(int16_t);
    break;
  case Q_I32:
    Q_ENC_I(int32_t);
    break;
  case Q_I64:
    Q_ENC_I(int64_t);
    break;
  case Q_F16: {
    const uint16_t *s = (const uint16_t *)src;
    int16_t *d = (int16_t *)dst;
    for (size_t i = 0; i < nbElts; ++i)
      d[i] = (int16_t)quant_index_fit(
          nearbyint((double)quant_half_to_float(s[i]) / step),
          quant_index_lo(Q_F16), quant_index_hi(Q_F16));
    break;
  }
  case Q_F32:
    Q_ENC_LIN_F(float, int32_t);
    break;
  case Q_F64:
    Q_ENC_LIN_F(double, int64_t);
    break;
  }
  return 0;
}

#define Q_SCAN(RD)                                                             \
  do {                                                                         \
    for (size_t i = 0; i < nbElts; ++i) {                                      \
      const double v = (double)(RD);                                           \
      if (!isfinite(v))                                                        \
        continue;                                                              \
      if (v < 0.0)                                                             \
        neg = 1;                                                               \
      const double a = fabs(v);                                                \
      if (a == 0.0)                                                            \
        continue;                                                              \
      if (a < lo)                                                              \
        lo = a;                                                                \
      if (a > hi)                                                              \
        hi = a;                                                                \
    }                                                                          \
  } while (0)

int quant_scan(const void *src, int dtype, size_t nbElts, double *minAbs,
               double *maxAbs, int *anyNegative) {
  double lo = INFINITY, hi = 0.0;
  int neg = 0;
  if (dtype < Q_U8 || dtype > Q_F64)
    return 1;
  switch ((quant_dtype)dtype) {
  case Q_U8:
    Q_SCAN(((const uint8_t *)src)[i]);
    break;
  case Q_U16:
    Q_SCAN(((const uint16_t *)src)[i]);
    break;
  case Q_U32:
    Q_SCAN(((const uint32_t *)src)[i]);
    break;
  case Q_U64:
    Q_SCAN(((const uint64_t *)src)[i]);
    break;
  case Q_I8:
    Q_SCAN(((const int8_t *)src)[i]);
    break;
  case Q_I16:
    Q_SCAN(((const int16_t *)src)[i]);
    break;
  case Q_I32:
    Q_SCAN(((const int32_t *)src)[i]);
    break;
  case Q_I64:
    Q_SCAN(((const int64_t *)src)[i]);
    break;
  case Q_F16:
    Q_SCAN(quant_half_to_float(((const uint16_t *)src)[i]));
    break;
  case Q_F32:
    Q_SCAN(((const float *)src)[i]);
    break;
  case Q_F64:
    Q_SCAN(((const double *)src)[i]);
    break;
  }
  if (minAbs)
    *minAbs = lo;
  if (maxAbs)
    *maxAbs = hi;
  if (anyNegative)
    *anyNegative = neg;
  return hi > 0.0 ? 0 : 1;
}