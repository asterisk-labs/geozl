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

#define QL_DEC_MUL(WT, IT, CAST)                                               \
  do {                                                                         \
    const IT *s = (const IT *)src;                                             \
    WT *d = (WT *)dst;                                                         \
    for (size_t i = 0; i < nbElts; ++i)                                        \
      d[i] = (WT)CAST(ql_clamp((double)s[i] * step, vlo, vhi));                \
  } while (0)

// The output range is wider than the whole stream range at every type, 32767
// against 65504 for a half and further apart above, so the only end of ql_clamp
// a stored reconstruction reaches is the floor at zero. The test is hoisted, a
// select the loop carries stops the vectoriser.
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

int quant_linear_decode(void *dst, const void *src,
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
