#include "decode_quant_linear_kernel.h"

#include "quant_linear_dtype.h"
#include "quant_linear_half.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

// Two selects and no branch, which is what lets gcc vectorise the loops below.
// An infinity lands on the end it belongs at.
static double ql_clamp(double v, double lo, double hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// Clamping toward an interval that holds x can only shorten |x - x^|, so the bound
// survives the floor.
static double ql_floor(const quant_linear_params *p, int dtype) {
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

#define QL_DEC_CAST(WT, IT, CAST)                                              \
  do {                                                                         \
    const IT *s = (const IT *)src;                                             \
    WT *d = (WT *)dst;                                                         \
    for (size_t i = 0; i < nbElts; ++i)                                        \
      d[i] = (WT)CAST(ql_clamp((double)s[i], vlo, vhi));                       \
  } while (0)

int quant_linear_decode(void *dst, const void *src,
                        const quant_linear_params *p, int dtype,
                        size_t nbElts) {
  // isfinite is what the clamp above leans on, since an infinite step turns a
  // zero sample into a NaN. Checked once here rather than once per sample.
  if (!QL_DTYPE_OK(dtype) || !isfinite(p->step) || !(p->step > 0.0))
    return 1;
  const double step = p->step;
  const double vlo = ql_floor(p, dtype), vhi = quant_linear_value_hi(dtype);
  const int values = (p->flags & QUANT_LINEAR_FLAG_STORE_VALUES) != 0;

  if (dtype <= QL_LAST_INT) {
    if (!values)
      return 1;
    memcpy(dst, src, nbElts * quant_linear_width(dtype));
    return 0;
  }

  // A whole number in an integer stream, so rebuilding is a cast. Exact for
  // everything the resolver let through, which is why this path owes the step
  // nothing for storage rounding.
  if (values) {
    if (floor(step) != step)
      return 1;
    switch ((ql_dtype)dtype) {
    case QL_F16: {
      const int16_t *s = (const int16_t *)src;
      uint16_t *d = (uint16_t *)dst;
      for (size_t i = 0; i < nbElts; ++i)
        d[i] = quant_linear_float_to_half(
            (float)ql_clamp((double)s[i], vlo, vhi));
      break;
    }
    case QL_F32:
      QL_DEC_CAST(float, int32_t, (float));
      break;
    default:
      QL_DEC_CAST(double, int64_t, );
      break;
    }
    return 0;
  }

  switch ((ql_dtype)dtype) {
  case QL_F16: {
    const int16_t *s = (const int16_t *)src;
    uint16_t *d = (uint16_t *)dst;
    for (size_t i = 0; i < nbElts; ++i)
      d[i] = quant_linear_float_to_half(
          (float)ql_clamp((double)s[i] * step, vlo, vhi));
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