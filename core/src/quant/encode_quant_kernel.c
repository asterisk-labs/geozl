#include "encode_quant_kernel.h"
#include "quant_half.h" // quant_half_to_float

#include <stdint.h>
#include <stdlib.h>
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
//
// On an integer input the map from value to index is a function of a finite
// set, so it goes in a table built once and the encode becomes a gather with no
// transcendental in it. The span comes from the data and not from the type: a
// uint32 raster of elevations covers a few thousand values, not four billion.
// The table and the loop below call the same function, so they cannot disagree.
#define Q_FWD_TABLE_MAX (1 << 18)
#define Q_FWD_TABLE_MIN 4

// Range of an integer stream. Unsigned throughout, so a span wider than a
// signed type cannot trap, and one that wrapped to zero fails the test that
// sizes the table rather than sizing it wrong.
#define Q_RANGE(T)                                                             \
  do {                                                                         \
    const T *v = (const T *)src;                                               \
    int64_t mn = (int64_t)v[0], mx = mn;                                       \
    for (size_t i = 1; i < nbElts; ++i) {                                      \
      const int64_t x = (int64_t)v[i];                                         \
      if (x < mn)                                                              \
        mn = x;                                                                \
      if (x > mx)                                                              \
        mx = x;                                                                \
    }                                                                          \
    *lo = mn;                                                                  \
    *n = (uint64_t)mx - (uint64_t)mn + 1u;                                     \
  } while (0)

static void q_int_range(const void *src, int dtype, size_t nbElts, int64_t *lo,
                        uint64_t *n) {
  *lo = 0;
  *n = 0;
  if (nbElts == 0)
    return;
  switch (dtype) {
  case Q_U8:
    Q_RANGE(uint8_t);
    break;
  case Q_U16:
    Q_RANGE(uint16_t);
    break;
  case Q_U32:
    Q_RANGE(uint32_t);
    break;
  case Q_I8:
    Q_RANGE(int8_t);
    break;
  case Q_I16:
    Q_RANGE(int16_t);
    break;
  case Q_I32:
    Q_RANGE(int32_t);
    break;
  case Q_I64:
    Q_RANGE(int64_t);
    break;
  default:
    break; // u64 does not fit an int64 offset, it keeps the direct path
  }
}

static double q_best_int(double x, const quant_params *p, double vlo,
                         double vhi, double ilo, double ihi) {
  double q = quant_fwd(x, p);
  const double r0 = QUANT_RT_INT(quant_inv(q, p), vlo, vhi);
  const double alt = q + (x > r0 ? 1.0 : -1.0);
  if (fabs(x - QUANT_RT_INT(quant_inv(alt, p), vlo, vhi)) < fabs(x - r0))
    q = alt;
  return quant_index_fit(q, ilo, ihi);
}

#define Q_ENC_WARP_INT(T)                                                      \
  do {                                                                         \
    const T *s = (const T *)src;                                               \
    T *d = (T *)dst;                                                           \
    if (tab != NULL) {                                                         \
      for (size_t i = 0; i < nbElts; ++i)                                      \
        d[i] = (T)tab[(size_t)((int64_t)s[i] - domLo)];                        \
    } else {                                                                   \
      for (size_t i = 0; i < nbElts; ++i)                                      \
        d[i] = (T)q_best_int((double)s[i], p, vlo, vhi, ilo, ihi);             \
    }                                                                          \
  } while (0)
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
    memcpy(dst, src, nbElts * quant_width(dtype));
    return 0;
  }

  if (p->curve != QUANT_CURVE_LINEAR) {
    const double vlo = quant_floor(p, dtype), vhi = quant_value_hi(dtype);
    const double ilo = quant_index_lo(dtype), ihi = quant_index_hi(dtype);

    // The table is per call, not cached, so two threads encoding at once do not
    // share it. An allocation that fails is not an error, the direct path below
    // gives the same answer.
    int64_t domLo = 0;
    uint64_t domN = 0;
    int64_t *tab = NULL;
    if (dtype <= Q_LAST_INT) {
      q_int_range(src, dtype, nbElts, &domLo, &domN);
      if (domN != 0 && domN <= Q_FWD_TABLE_MAX &&
          nbElts >= Q_FWD_TABLE_MIN * domN)
        tab = (int64_t *)malloc((size_t)domN * sizeof(int64_t));
      if (tab != NULL)
        for (uint64_t v = 0; v < domN; ++v)
          tab[v] = (int64_t)q_best_int((double)(domLo + (int64_t)v), p, vlo,
                                       vhi, ilo, ihi);
    }

    switch ((quant_dtype)dtype) {
    case Q_U8:
      Q_ENC_WARP_INT(uint8_t);
      break;
    case Q_U16:
      Q_ENC_WARP_INT(uint16_t);
      break;
    case Q_U32:
      Q_ENC_WARP_INT(uint32_t);
      break;
    case Q_U64:
      Q_ENC_WARP(((const uint64_t *)src)[i], QUANT_RT_INT, uint64_t);
      break;
    case Q_I8:
      Q_ENC_WARP_INT(int8_t);
      break;
    case Q_I16:
      Q_ENC_WARP_INT(int16_t);
      break;
    case Q_I32:
      Q_ENC_WARP_INT(int32_t);
      break;
    case Q_I64:
      Q_ENC_WARP_INT(int64_t);
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
    free(tab);
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
          nearbyint((double)quant_half_to_float(s[i]) / step), ilo, ihi);
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