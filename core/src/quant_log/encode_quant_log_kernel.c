#include "encode_quant_log_kernel.h"

#include "quant_log_dtype.h"
#include "quant_log_half.h"
#include "quant_log_math.h"

#include <stdint.h>
#include <stdlib.h>

// Past this is an infinity, which pins to the top level rather than running the
// transform on an exponent field of all ones.
#define QLOG_FINITE_TOP 1.7976931348623157e308

// Zero is the one sample no level describes, and so is a NaN, which has no
// magnitude to place. Both go to zero and come back as zero. A tile that means
// its NaNs carries the nodata codec in front. The sign of a negative zero is not
// kept.

// The index is built as an integer in one step rather than rounded as a double
// and converted after. Neither transform is ever negative, so truncating the sum
// lands on the same level rounding would and costs one conversion instead of
// three.
//
// The range is folded in double before the conversion, never after. A step small
// enough drives the quotient past what an int64 holds, and the conversion is then
// undefined and does different things on different machines: x86 hands back the
// most negative value and arm64 saturates at the most positive one. The tests are
// written so a NaN, which a zero step and an infinite quotient can produce, falls
// to the floor rather than reaching the conversion.
static inline int64_t qlog_fit(double v, double lo, double top) {
  if (!(v > lo))
    return (int64_t)lo;
  if (!(v < top))
    return (int64_t)top;
  return (int64_t)v;
}

#define QLOG_PICK(a, anchor)                                                   \
  ((a) > QLOG_FINITE_TOP                                                       \
       ? top                                                                   \
       : qlog_fit(qlog_log2_over((a), (anchor)) * inv + 0.5, 0.0, dtop))

// Cases 1 and 2, the value grid. The stream carries the reconstruction, so the
// table holds whole numbers already folded onto the output range.
#define QLOG_ENC_U(T)                                                          \
  do {                                                                         \
    const T *s = (const T *)src;                                               \
    T *d = (T *)dst;                                                           \
    for (size_t i = 0; i < nbElts; ++i) {                                      \
      const double a = (double)s[i];                                           \
      d[i] = a > 0.0 ? (T)lev[QLOG_PICK(a, 0)] : (T)0;                         \
    }                                                                          \
  } while (0)

#define QLOG_ENC_I(T)                                                          \
  do {                                                                         \
    const T *s = (const T *)src;                                               \
    T *d = (T *)dst;                                                           \
    for (size_t i = 0; i < nbElts; ++i) {                                      \
      const int64_t v = (int64_t)s[i];                                         \
      const double a = v < 0 ? -(double)v : (double)v;                         \
      if (!(a > 0.0)) {                                                        \
        d[i] = 0;                                                              \
        continue;                                                              \
      }                                                                        \
      const double r = lev[QLOG_PICK(a, 0)];                                   \
      d[i] = (T)(int64_t)(v < 0 ? -r : r);                                     \
    }                                                                          \
  } while (0)

#define QLOG_ENC_FV(WT, IT, RD)                                                \
  do {                                                                         \
    const WT *s = (const WT *)src;                                             \
    IT *d = (IT *)dst;                                                         \
    for (size_t i = 0; i < nbElts; ++i) {                                      \
      const double x = (double)(RD);                                           \
      const double a = x < 0.0 ? -x : x;                                       \
      if (!(a > 0.0)) {                                                        \
        d[i] = 0;                                                              \
        continue;                                                              \
      }                                                                        \
      const double r = lev[QLOG_PICK(a, 0)];                                   \
      d[i] = (IT)(int64_t)(x < 0.0 ? -r : r);                                  \
    }                                                                          \
  } while (0)

// Case 3, the index grid. Index magnitude one is the anchor, so zero is free.
#define QLOG_ENC_FI(WT, IT, RD)                                                \
  do {                                                                         \
    const WT *s = (const WT *)src;                                             \
    IT *d = (IT *)dst;                                                         \
    for (size_t i = 0; i < nbElts; ++i) {                                      \
      const double x = (double)(RD);                                           \
      const double a = x < 0.0 ? -x : x;                                       \
      if (!(a > 0.0)) {                                                        \
        d[i] = 0;                                                              \
        continue;                                                              \
      }                                                                        \
      const int64_t m =                                                        \
          a > QLOG_FINITE_TOP                                                  \
              ? top                                                            \
              : qlog_fit(qlog_log2_over(a, anchor) * inv + 1.5, 1.0, dtop);    \
      d[i] = (IT)(x < 0.0 ? -m : m);                                           \
    }                                                                          \
  } while (0)

int quant_log_encode(void *dst, const void *src, const quant_log_params *p,
                     int dtype, size_t nbElts) {
  if (!QLOG_DTYPE_OK(dtype) || !(p->step > 0.0))
    return 1;
  const int values = (p->flags & QUANT_LOG_FLAG_STORE_VALUES) != 0;
  if ((dtype <= QLOG_LAST_INT) != (values && dtype <= QLOG_LAST_INT))
    return 1; // an integer type carries the reconstruction and nothing else
  const double inv = 1.0 / p->step;

  if (!values) {
    const int anchor = quant_log_anchor_exp2(dtype);
    const double dtop = quant_log_index_top(p->step, dtype);
    const int64_t top = (int64_t)dtop;
    switch ((qlog_dtype)dtype) {
    case QLOG_F16:
      QLOG_ENC_FI(uint16_t, int16_t, quant_log_half_to_float(s[i]));
      break;
    case QLOG_F32:
      QLOG_ENC_FI(float, int32_t, s[i]);
      break;
    case QLOG_F64:
      QLOG_ENC_FI(double, int64_t, s[i]);
      break;
    default:
      return 1;
    }
    return 0;
  }

  const int n = quant_log_level_count(p->step, dtype);
  if (n <= 0)
    return 1;
  double *lev = (double *)malloc((size_t)n * sizeof(double));
  if (lev == NULL)
    return 1;
  const int64_t top = n - 1;
  const double dtop = (double)top;
  const double cap = dtype <= QLOG_LAST_INT ? quant_log_value_hi(dtype)
                                            : quant_log_exact_int(dtype);
  for (int j = 0; j < n; ++j) {
    const double v = qlog_value_level((double)j, p->step);
    lev[j] = v > cap ? cap : v;
  }

  switch ((qlog_dtype)dtype) {
  case QLOG_U8:
    QLOG_ENC_U(uint8_t);
    break;
  case QLOG_U16:
    QLOG_ENC_U(uint16_t);
    break;
  case QLOG_U32:
    QLOG_ENC_U(uint32_t);
    break;
  case QLOG_U64:
    QLOG_ENC_U(uint64_t);
    break;
  case QLOG_I8:
    QLOG_ENC_I(int8_t);
    break;
  case QLOG_I16:
    QLOG_ENC_I(int16_t);
    break;
  case QLOG_I32:
    QLOG_ENC_I(int32_t);
    break;
  case QLOG_I64:
    QLOG_ENC_I(int64_t);
    break;
  case QLOG_F16:
    QLOG_ENC_FV(uint16_t, int16_t, quant_log_half_to_float(s[i]));
    break;
  case QLOG_F32:
    QLOG_ENC_FV(float, int32_t, s[i]);
    break;
  default:
    QLOG_ENC_FV(double, int64_t, s[i]);
    break;
  }
  free(lev);
  return 0;
}

#define QLOG_SCAN(RD)                                                          \
  do {                                                                         \
    for (size_t i = 0; i < nbElts; ++i) {                                      \
      const double v = (double)(RD);                                           \
      if (v != v)                                                              \
        continue;                                                              \
      if (v < 0.0)                                                             \
        neg = 1;                                                               \
      const double a = v < 0.0 ? -v : v;                                       \
      if (a == 0.0 || a > QLOG_FINITE_TOP)                                     \
        continue;                                                              \
      if (a < lo)                                                              \
        lo = a;                                                                \
      if (a > hi)                                                              \
        hi = a;                                                                \
      if (a < nmin)                                                            \
        sub = 1;                                                               \
    }                                                                          \
  } while (0)

int quant_log_scan(const void *src, int dtype, size_t nbElts,
                   quant_log_stats *out) {
  double lo = 1.0e308 * 10.0, hi = 0.0; // lo starts at inf
  int neg = 0, sub = 0;
  if (!QLOG_DTYPE_OK(dtype) || out == NULL)
    return 1;
  const double nmin =
      dtype > QLOG_LAST_INT ? quant_log_normal_min(dtype) : 0.0;

  switch ((qlog_dtype)dtype) {
  case QLOG_U8:
    QLOG_SCAN(((const uint8_t *)src)[i]);
    break;
  case QLOG_U16:
    QLOG_SCAN(((const uint16_t *)src)[i]);
    break;
  case QLOG_U32:
    QLOG_SCAN(((const uint32_t *)src)[i]);
    break;
  case QLOG_U64:
    QLOG_SCAN(((const uint64_t *)src)[i]);
    break;
  case QLOG_I8:
    QLOG_SCAN(((const int8_t *)src)[i]);
    break;
  case QLOG_I16:
    QLOG_SCAN(((const int16_t *)src)[i]);
    break;
  case QLOG_I32:
    QLOG_SCAN(((const int32_t *)src)[i]);
    break;
  case QLOG_I64:
    QLOG_SCAN(((const int64_t *)src)[i]);
    break;
  case QLOG_F16:
    QLOG_SCAN(quant_log_half_to_float(((const uint16_t *)src)[i]));
    break;
  case QLOG_F32:
    QLOG_SCAN(((const float *)src)[i]);
    break;
  default:
    QLOG_SCAN(((const double *)src)[i]);
    break;
  }

  out->minAbs = lo;
  out->maxAbs = hi;
  out->anyNegative = neg;
  out->anySubnormal = sub;
  return 0;
}
