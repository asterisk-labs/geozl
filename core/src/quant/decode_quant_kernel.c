#include "decode_quant_kernel.h"
#include "quant_half.h" // quant_float_to_half

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// The linear curve keeps the exact integer arithmetic it always had. The warped
// curves rebuild through quant_inv, which costs an exp per element on the log
// curve; the table below removes it whenever the index range is small enough,
// which on the data these curves are meant for it always is.

#define Q_DEC_U(T)                                                             \
  do {                                                                         \
    const T *s = (const T *)src;                                               \
    T *d = (T *)dst;                                                           \
    const uint64_t isc = (uint64_t)(step < 1.0 ? 1.0 : step);                  \
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
    const int64_t isc = (int64_t)(step < 1.0 ? 1.0 : step);                    \
    for (size_t i = 0; i < nbElts; ++i) {                                      \
      __int128 r = (__int128)s[i] * isc;                                       \
      if (r < (LO))                                                            \
        r = (LO);                                                              \
      if (r > (HI))                                                            \
        r = (HI);                                                              \
      d[i] = (T)(int64_t)r;                                                    \
    }                                                                          \
  } while (0)

#define Q_DEC_LIN_F(WT, IT)                                                    \
  do {                                                                         \
    const IT *s = (const IT *)src;                                             \
    WT *d = (WT *)dst;                                                         \
    for (size_t i = 0; i < nbElts; ++i)                                        \
      d[i] = (WT)((double)s[i] * step);                                        \
  } while (0)

// Reconstruct through a table of one entry per index when the range allows it,
// so the log curve does not pay a transcendental per sample. Above the cap the
// table would leave L1 and stop being worth the pass that builds it.
#define Q_LUT_MAX 8192

#define Q_DEC_WARP(WT, IT, ST)                                                 \
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
    const int64_t span = hi - lo + 1;                                          \
    if (nbElts != 0 && span > 0 && span <= Q_LUT_MAX)                          \
      lut = (WT *)malloc((size_t)span * sizeof(WT));                           \
    if (lut != NULL) {                                                         \
      for (int64_t k = 0; k < span; ++k)                                       \
        lut[k] = (WT)(ST)quant_inv((double)(lo + k), p);                       \
      for (size_t i = 0; i < nbElts; ++i)                                      \
        d[i] = lut[(int64_t)s[i] - lo];                                        \
      free(lut);                                                               \
    } else {                                                                   \
      for (size_t i = 0; i < nbElts; ++i)                                      \
        d[i] = (WT)(ST)quant_inv((double)s[i], p);                             \
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
    switch ((quant_dtype)dtype) {
    case Q_U8:
      Q_DEC_WARP(uint8_t, uint8_t, uint64_t);
      break;
    case Q_U16:
      Q_DEC_WARP(uint16_t, uint16_t, uint64_t);
      break;
    case Q_U32:
      Q_DEC_WARP(uint32_t, uint32_t, uint64_t);
      break;
    case Q_U64:
      Q_DEC_WARP(uint64_t, uint64_t, uint64_t);
      break;
    case Q_I8:
      Q_DEC_WARP(int8_t, int8_t, int64_t);
      break;
    case Q_I16:
      Q_DEC_WARP(int16_t, int16_t, int64_t);
      break;
    case Q_I32:
      Q_DEC_WARP(int32_t, int32_t, int64_t);
      break;
    case Q_I64:
      Q_DEC_WARP(int64_t, int64_t, int64_t);
      break;
    case Q_F16: {
      const int16_t *s = (const int16_t *)src;
      uint16_t *d = (uint16_t *)dst;
      for (size_t i = 0; i < nbElts; ++i)
        d[i] = quant_float_to_half((float)quant_inv((double)s[i], p));
      break;
    }
    case Q_F32:
      Q_DEC_WARP(float, int32_t, float);
      break;
    case Q_F64:
      Q_DEC_WARP(double, int64_t, double);
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
      d[i] = quant_float_to_half((float)((double)s[i] * step));
    break;
  }
  case Q_F32:
    Q_DEC_LIN_F(float, int32_t);
    break;
  case Q_F64:
    Q_DEC_LIN_F(double, int64_t);
    break;
  }
  return 0;
}
