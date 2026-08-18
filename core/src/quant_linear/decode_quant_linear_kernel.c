#include "decode_quant_linear_kernel.h"

#include "quant_linear_check.h"
#include "quant_linear_dtype.h"
#include "quant_linear_half.h"

#include <stdint.h>
#include <string.h>

// Two selects and no branch, so the index loops below vectorise.
static double ql_clamp(double v, double lo, double hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// Clamping into an interval that holds x can only shorten the error.
static double ql_lo(const quant_linear_params *p, int dtype) {
  return (p->flags & QUANT_LINEAR_FLAG_NONNEGATIVE) != 0
             ? 0.0
             : quant_linear_value_lo(dtype);
}

// Saturate before multiplying; a forged wide index can overflow 128 bits.
#define QL_DEC_IDX_U(T)                                                        \
  do {                                                                         \
    const T *s = (const T *)src;                                               \
    T *d = (T *)dst;                                                           \
    const unsigned __int128 cap = (unsigned __int128)(T)(~(T)0);               \
    const unsigned __int128 lim = cap / isc;                                   \
    for (size_t i = 0; i < nbElts; ++i) {                                      \
      const unsigned __int128 m = (unsigned __int128)s[i];                     \
      d[i] = (T)(uint64_t)(m > lim ? cap : m * isc);                           \
    }                                                                          \
  } while (0)

// TYPE_MIN has magnitude HI+1, so signed saturation uses that upper bound.
#define QL_DEC_IDX_I(T, LO, HI)                                                \
  do {                                                                         \
    const T *s = (const T *)src;                                               \
    T *d = (T *)dst;                                                           \
    const T lo = floorZero ? (T)0 : (T)(LO);                                   \
    const unsigned __int128 top = (unsigned __int128)(HI) + 1u;                \
    const unsigned __int128 lim = top / isc;                                   \
    for (size_t i = 0; i < nbElts; ++i) {                                      \
      const __int128 q = (__int128)s[i];                                       \
      const unsigned __int128 m =                                              \
          q < 0 ? (unsigned __int128)(-q) : (unsigned __int128)q;              \
      const unsigned __int128 mr = m > lim ? top : m * isc;                    \
      const __int128 r = q < 0 ? -(__int128)mr : (__int128)mr;                 \
      d[i] = r < (__int128)lo ? lo : (r > (__int128)(HI) ? (HI) : (T)r);       \
    }                                                                          \
  } while (0)

#define QL_DEC_MUL(WT, IT, CAST)                                               \
  do {                                                                         \
    const IT *s = (const IT *)src;                                             \
    WT *d = (WT *)dst;                                                         \
    for (size_t i = 0; i < nbElts; ++i)                                        \
      d[i] = (WT)CAST(ql_clamp((double)s[i] * step, vlo, vhi));                \
  } while (0)

// A stored reconstruction is already inside its output type, so the floor at zero
// is the only part of ql_clamp it can reach. The test is hoisted, a select the
// loop carries stops the vectoriser.
#define QL_DEC_CAST(WT, IT, CONV)                                              \
  do {                                                                         \
    const IT *s = (const IT *)src;                                             \
    WT *d = (WT *)dst;                                                         \
    if (floorZero)                                                             \
      for (size_t i = 0; i < nbElts; ++i) {                                    \
        const IT v = s[i];                                                     \
        d[i] = CONV(v < 0 ? (IT)0 : v);                                        \
      }                                                                        \
    else                                                                       \
      for (size_t i = 0; i < nbElts; ++i)                                      \
        d[i] = CONV(s[i]);                                                     \
  } while (0)

int quant_linear_decode(void *restrict dst, const void *restrict src,
                        const quant_linear_params *p, int dtype,
                        size_t nbElts) {
  // Same predicate the encoder reads. The clamp leans on a finite step.
  if (!quant_linear_params_ok(p, dtype))
    return 1;

  const double step = p->step;
  const double vlo = ql_lo(p, dtype), vhi = quant_linear_value_hi(dtype);
  const int values = (p->flags & QUANT_LINEAR_FLAG_STORE_VALUES) != 0;
  const int floorZero = (p->flags & QUANT_LINEAR_FLAG_NONNEGATIVE) != 0;

  if (dtype <= QL_LAST_INT) {
    // Rebuild integer indices with the encoder's whole step.
    if (!values) {
      const uint64_t isc = quant_linear_step_u64(step);
      switch ((ql_dtype)dtype) {
      case QL_U8:
        QL_DEC_IDX_U(uint8_t);
        break;
      case QL_U16:
        QL_DEC_IDX_U(uint16_t);
        break;
      case QL_U32:
        QL_DEC_IDX_U(uint32_t);
        break;
      case QL_U64:
        QL_DEC_IDX_U(uint64_t);
        break;
      case QL_I8:
        QL_DEC_IDX_I(int8_t, INT8_MIN, INT8_MAX);
        break;
      case QL_I16:
        QL_DEC_IDX_I(int16_t, INT16_MIN, INT16_MAX);
        break;
      case QL_I32:
        QL_DEC_IDX_I(int32_t, INT32_MIN, INT32_MAX);
        break;
      default:
        QL_DEC_IDX_I(int64_t, INT64_MIN, INT64_MAX);
        break;
      }
      return 0;
    }
    // An unsigned type cannot hold a negative, so the floor is already true of
    // the stream and the copy stands. A signed one has to have it applied, the
    // same way the float STORE=VALUES paths below do. Without this the flag
    // means the floor on a float frame and nothing at all on an integer one.
    if (floorZero && dtype >= QL_I8) {
      switch ((ql_dtype)dtype) {
      case QL_I8:
        QL_DEC_CAST(int8_t, int8_t, (int8_t));
        break;
      case QL_I16:
        QL_DEC_CAST(int16_t, int16_t, (int16_t));
        break;
      case QL_I32:
        QL_DEC_CAST(int32_t, int32_t, (int32_t));
        break;
      default:
        QL_DEC_CAST(int64_t, int64_t, (int64_t));
        break;
      }
      return 0;
    }
    memcpy(dst, src, nbElts * quant_linear_width(dtype));
    return 0;
  }

  // A whole number in an integer stream, so rebuilding is a cast.
  if (values) {
    switch ((ql_dtype)dtype) {
    case QL_F16:
      QL_DEC_CAST(uint16_t, int16_t, ql_i16_to_half);
      break;
    case QL_F32:
      QL_DEC_CAST(float, int32_t, (float));
      break;
    default:
      QL_DEC_CAST(double, int64_t, (double));
      break;
    }
    return 0;
  }

  switch ((ql_dtype)dtype) {
  case QL_F16: {
    const int16_t *s = (const int16_t *)src;
    uint16_t *d = (uint16_t *)dst;
    for (size_t i = 0; i < nbElts; ++i)
      d[i] = ql_f32_to_half((float)ql_clamp((double)s[i] * step, vlo, vhi));
    break;
  }
  case QL_F32:
    QL_DEC_MUL(float, int32_t, (float));
    break;
  default:
    QL_DEC_MUL(double, int64_t, );
    break;
  }
  return 0;
}

#undef QL_DEC_MUL
#undef QL_DEC_CAST
