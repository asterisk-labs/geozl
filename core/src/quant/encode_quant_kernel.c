#include "encode_quant_kernel.h"
#include "quant_half.h" // quant_half_to_float

#include <stdint.h>
#include <string.h>

// The linear curve keeps exact integer arithmetic, so a 64 bit sample does not
// lose precision through a double, and its paths are unchanged from what the
// codec did before the curves were added. The warped curves go through double,
// which costs nothing on the float types they are meant for.

#define Q_ENC_U(T)                                                             \
  do {                                                                         \
    const T *s = (const T *)src;                                               \
    T *d = (T *)dst;                                                           \
    const uint64_t isc = (uint64_t)(step < 1.0 ? 1.0 : step);                  \
    const uint64_t half = isc >> 1;                                            \
    for (size_t i = 0; i < nbElts; ++i)                                        \
      d[i] = (T)(((unsigned __int128)s[i] + half) / isc);                      \
  } while (0)

#define Q_ENC_I(T)                                                             \
  do {                                                                         \
    const T *s = (const T *)src;                                               \
    T *d = (T *)dst;                                                           \
    const int64_t isc = (int64_t)(step < 1.0 ? 1.0 : step);                    \
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
    const uint64_t isc = (uint64_t)(step < 1.0 ? 1.0 : step);                  \
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
    const int64_t isc = (int64_t)(step < 1.0 ? 1.0 : step);                    \
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

#define Q_ENC_LIN_F(WT, IT, MINV, MAXV)                                        \
  do {                                                                         \
    const WT *s = (const WT *)src;                                             \
    IT *d = (IT *)dst;                                                         \
    for (size_t i = 0; i < nbElts; ++i) {                                      \
      double q = nearbyint((double)s[i] / step);                               \
      if (q < (double)(MINV))                                                  \
        q = (double)(MINV);                                                    \
      if (q > (double)(MAXV))                                                  \
        q = (double)(MAXV);                                                    \
      d[i] = (IT)q;                                                            \
    }                                                                          \
  } while (0)

// The warped curves pick the index whose reconstruction, taken back at the
// output width, is nearest the sample. quant_fwd already lands on it or one
// short, and the curves are monotone, so only the neighbour on the side of the
// residual can be closer. Testing it is what keeps the declared bound from
// depending on how accurate log and exp happen to be.
#define Q_RT_INT(v) nearbyint(v)
#define Q_RT_F16(v) ((double)quant_half_to_float(quant_float_to_half((float)(v))))
#define Q_RT_F32(v) ((double)(float)(v))
#define Q_RT_F64(v) (v)

#define Q_ENC_WARP(RD, RT, IT, MINV, MAXV)                                     \
  do {                                                                         \
    IT *d = (IT *)dst;                                                         \
    for (size_t i = 0; i < nbElts; ++i) {                                      \
      const double x = (double)(RD);                                           \
      double q = quant_fwd(x, p);                                              \
      const double r0 = RT(quant_inv(q, p));                                   \
      const double alt = q + (x > r0 ? 1.0 : -1.0);                            \
      if (fabs(x - RT(quant_inv(alt, p))) < fabs(x - r0))                      \
        q = alt;                                                               \
      if (q < (double)(MINV))                                                  \
        q = (double)(MINV);                                                    \
      if (q > (double)(MAXV))                                                  \
        q = (double)(MAXV);                                                    \
      d[i] = (IT)q;                                                            \
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
    switch ((quant_dtype)dtype) {
    case Q_U8:
      Q_ENC_WARP(((const uint8_t *)src)[i], Q_RT_INT, uint8_t, 0, UINT8_MAX);
      break;
    case Q_U16:
      Q_ENC_WARP(((const uint16_t *)src)[i], Q_RT_INT, uint16_t, 0, UINT16_MAX);
      break;
    case Q_U32:
      Q_ENC_WARP(((const uint32_t *)src)[i], Q_RT_INT, uint32_t, 0, UINT32_MAX);
      break;
    case Q_U64:
      Q_ENC_WARP(((const uint64_t *)src)[i], Q_RT_INT, uint64_t, 0,
                 18446744073709549568.0);
      break;
    case Q_I8:
      Q_ENC_WARP(((const int8_t *)src)[i], Q_RT_INT, int8_t, INT8_MIN, INT8_MAX);
      break;
    case Q_I16:
      Q_ENC_WARP(((const int16_t *)src)[i], Q_RT_INT, int16_t, INT16_MIN, INT16_MAX);
      break;
    case Q_I32:
      Q_ENC_WARP(((const int32_t *)src)[i], Q_RT_INT, int32_t, INT32_MIN, INT32_MAX);
      break;
    case Q_I64:
      Q_ENC_WARP(((const int64_t *)src)[i], Q_RT_INT, int64_t, INT64_MIN,
                 9223372036854774784.0);
      break;
    case Q_F16:
      Q_ENC_WARP(quant_half_to_float(((const uint16_t *)src)[i]), Q_RT_F16,
                 int16_t, INT16_MIN, INT16_MAX);
      break;
    case Q_F32:
      Q_ENC_WARP(((const float *)src)[i], Q_RT_F32, int32_t, INT32_MIN,
                 INT32_MAX);
      break;
    case Q_F64:
      Q_ENC_WARP(((const double *)src)[i], Q_RT_F64, int64_t, INT64_MIN,
                 9223372036854774784.0);
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
    for (size_t i = 0; i < nbElts; ++i) {
      double q = nearbyint((double)quant_half_to_float(s[i]) / step);
      if (q < -32768.0)
        q = -32768.0;
      if (q > 32767.0)
        q = 32767.0;
      d[i] = (int16_t)q;
    }
    break;
  }
  case Q_F32:
    Q_ENC_LIN_F(float, int32_t, INT32_MIN, INT32_MAX);
    break;
  case Q_F64:
    Q_ENC_LIN_F(double, int64_t, INT64_MIN, INT64_MAX);
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
