#include "decode_quant_kernel.h"
#include "quant_half.h" // quant_float_to_half

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// The linear curve stays on exact integer arithmetic. The warped curves rebuild
// through quant_inv, an exp per element on the log curve, which the table below
// takes out whenever the index range is small enough.

#define Q_DEC_U(T)                                                             \
  do {                                                                         \
    const T *s = (const T *)src;                                               \
    T *d = (T *)dst;                                                           \
    const uint64_t isc = quant_step_u64(step);                                 \
    const uint64_t cap = (uint64_t)(T)(~(T)0);                                 \
    for (size_t i = 0; i < nbElts; ++i) {                                      \
      const unsigned __int128 r = (unsigned __int128)s[i] * isc;               \
      d[i] = (T)(r < cap ? (uint64_t)r : cap);                                 \
    }                                                                          \
  } while (0)

#define Q_DEC_I(T, LO, HI)                                                     \
  do {                                                                         \
    const T *s = (const T *)src;                                               \
    T *d = (T *)dst;                                                           \
    const int64_t isc = quant_step_i64(step);                                  \
    for (size_t i = 0; i < nbElts; ++i) {                                      \
      __int128 r = (__int128)s[i] * isc;                                       \
      if (r < (LO))                                                            \
        r = (LO);                                                              \
      if (r > (HI))                                                            \
        r = (HI);                                                              \
      d[i] = (T)(int64_t)r;                                                    \
    }                                                                          \
  } while (0)

#define Q_DEC_LIN_F(WT, IT, VLO, VHI)                                          \
  do {                                                                         \
    const IT *s = (const IT *)src;                                             \
    WT *d = (WT *)dst;                                                         \
    for (size_t i = 0; i < nbElts; ++i)                                        \
      d[i] = (WT)quant_clamp((double)s[i] * step, (VLO), (VHI));               \
  } while (0)

// Above this the table leaves L1 and stops being worth the pass that builds it.
#define Q_LUT_MAX 8192

#define Q_DEC_WARP(WT, IT, RT)                                                 \
  do {                                                                         \
    const IT *s = (const IT *)src;                                             \
    WT *d = (WT *)dst;                                                         \
    int64_t lo = 0, hi = 0;                                                    \
    if (nbElts != 0) {                                                         \
      lo = hi = (int64_t)s[0];                                                 \
      for (size_t i = 1; i < nbElts; ++i) {                                    \
        const int64_t v = (int64_t)s[i];                                       \
        if (v < lo)                                                            \
          lo = v;                                                              \
        if (v > hi)                                                            \
          hi = v;                                                              \
      }                                                                        \
    }                                                                          \
    WT *lut = NULL;                                                            \
    /* hi - lo overflows a signed type on a forged stream. Unsigned wraps      \
       instead, and a span that wrapped to zero fails the test below rather    \
       than sizing a table the loop then reads past. */                        \
    const uint64_t span = (uint64_t)hi - (uint64_t)lo + 1u;                    \
    if (nbElts != 0 && span != 0 && span <= Q_LUT_MAX)                         \
      lut = (WT *)malloc((size_t)span * sizeof(WT));                           \
    if (lut != NULL) {                                                         \
      for (uint64_t k = 0; k < span; ++k)                                      \
        lut[k] = (WT)RT(quant_inv((double)(lo + (int64_t)k), p), vlo, vhi);    \
      for (size_t i = 0; i < nbElts; ++i)                                      \
        d[i] = lut[(uint64_t)(int64_t)s[i] - (uint64_t)lo];                    \
      free(lut);                                                               \
    } else {                                                                   \
      for (size_t i = 0; i < nbElts; ++i)                                      \
        d[i] = (WT)RT(quant_inv((double)s[i], p), vlo, vhi);                   \
    }                                                                          \
  } while (0)

int quant_decode(void *dst, const void *src, const quant_params *p, int dtype,
                 size_t nbElts) {
  if (dtype < Q_U8 || dtype > Q_F64)
    return 1;
  if (p->curve > QUANT_CURVE_LOG)
    return 1;

  double step = p->step;
  // A stored reconstruction is already the answer, and so are a step of zero
  // and, on integers under the linear curve, a step of one.
  const int copy = (p->flags & QUANT_FLAG_STORE_VALUES) != 0 || step == 0.0 ||
                   (p->curve == QUANT_CURVE_LINEAR && step == 1.0 &&
                    dtype <= Q_LAST_INT);
  if (copy) {
    static const size_t w[] = {1, 2, 4, 8, 1, 2, 4, 8, 2, 4, 8};
    memcpy(dst, src, nbElts * w[dtype]);
    return 0;
  }

  if (p->curve != QUANT_CURVE_LINEAR) {
    const double vlo = quant_value_lo(dtype), vhi = quant_value_hi(dtype);
    switch ((quant_dtype)dtype) {
    case Q_U8:
      Q_DEC_WARP(uint8_t, uint8_t, QUANT_RT_INT);
      break;
    case Q_U16:
      Q_DEC_WARP(uint16_t, uint16_t, QUANT_RT_INT);
      break;
    case Q_U32:
      Q_DEC_WARP(uint32_t, uint32_t, QUANT_RT_INT);
      break;
    case Q_U64:
      Q_DEC_WARP(uint64_t, uint64_t, QUANT_RT_INT);
      break;
    case Q_I8:
      Q_DEC_WARP(int8_t, int8_t, QUANT_RT_INT);
      break;
    case Q_I16:
      Q_DEC_WARP(int16_t, int16_t, QUANT_RT_INT);
      break;
    case Q_I32:
      Q_DEC_WARP(int32_t, int32_t, QUANT_RT_INT);
      break;
    case Q_I64:
      Q_DEC_WARP(int64_t, int64_t, QUANT_RT_INT);
      break;
    case Q_F16: {
      const int16_t *s = (const int16_t *)src;
      uint16_t *d = (uint16_t *)dst;
      for (size_t i = 0; i < nbElts; ++i)
        d[i] = quant_float_to_half(
            (float)QUANT_RT_F16(quant_inv((double)s[i], p), vlo, vhi));
      break;
    }
    case Q_F32:
      Q_DEC_WARP(float, int32_t, QUANT_RT_F32);
      break;
    case Q_F64:
      Q_DEC_WARP(double, int64_t, QUANT_RT_F64);
      break;
    }
    return 0;
  }

  switch ((quant_dtype)dtype) {
  case Q_U8:
    Q_DEC_U(uint8_t);
    break;
  case Q_U16:
    Q_DEC_U(uint16_t);
    break;
  case Q_U32:
    Q_DEC_U(uint32_t);
    break;
  case Q_U64:
    Q_DEC_U(uint64_t);
    break;
  case Q_I8:
    Q_DEC_I(int8_t, INT8_MIN, INT8_MAX);
    break;
  case Q_I16:
    Q_DEC_I(int16_t, INT16_MIN, INT16_MAX);
    break;
  case Q_I32:
    Q_DEC_I(int32_t, INT32_MIN, INT32_MAX);
    break;
  case Q_I64:
    Q_DEC_I(int64_t, INT64_MIN, INT64_MAX);
    break;
  case Q_F16: {
    const int16_t *s = (const int16_t *)src;
    uint16_t *d = (uint16_t *)dst;
    for (size_t i = 0; i < nbElts; ++i)
      d[i] = quant_float_to_half((float)quant_clamp(
          (double)s[i] * step, quant_value_lo(Q_F16), quant_value_hi(Q_F16)));
    break;
  }
  case Q_F32:
    Q_DEC_LIN_F(float, int32_t, quant_value_lo(Q_F32), quant_value_hi(Q_F32));
    break;
  case Q_F64:
    Q_DEC_LIN_F(double, int64_t, quant_value_lo(Q_F64), quant_value_hi(Q_F64));
    break;
  }
  return 0;
}