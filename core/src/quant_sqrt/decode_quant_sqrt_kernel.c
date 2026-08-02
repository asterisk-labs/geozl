#include "decode_quant_sqrt_kernel.h"

#include "quant_sqrt_dtype.h"
#include "quant_sqrt_half.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

// The rebuild is (q*step)^2 - offset, two multiplies and a subtract, so the whole
// loop stays in registers and runs at the width of the vector unit. Two things
// below are what keep it there, and both look like details.
//
// The index is folded before any arithmetic, so what comes out is finite and in
// range by construction and the loop never has to test for a NaN. A NaN test is
// control flow, and control flow anywhere in the body is enough for gcc to give
// up on the loop.
//
// The fold is one unsigned min rather than a pair of signed compares. The encoder
// never emits a negative index, since the grid lives over x + offset >= 0, so one
// out of a damaged frame is nonsense either way and reading it unsigned sends it
// to the top of the grid rather than through a second branch. It also sidesteps a
// gcc quirk: a lower bound written as a literal zero, or as a const that folds to
// one, turns the expression into a branch and the loop is refused. Checked on gcc
// 13.3 with -fopt-info-vec.

#define QSQ_DEC(WT, IT, UT, ATYPE, STORE)                                      \
  do {                                                                         \
    const IT *s = (const IT *)src;                                             \
    WT *d = (WT *)dst;                                                         \
    const ATYPE st = (ATYPE)step, of = (ATYPE)offset;                          \
    const ATYPE lo = (ATYPE)vlo, hi = (ATYPE)vhi;                              \
    for (size_t i = 0; i < nbElts; ++i) {                                      \
      UT u = (UT)s[i];                                                         \
      u = u > utop ? utop : u;                                                 \
      const ATYPE t = (ATYPE)(IT)u * st;                                       \
      ATYPE v = t * t - of;                                                    \
      v = v < lo ? lo : (v > hi ? hi : v);                                     \
      d[i] = (WT)(STORE(v));                                                   \
    }                                                                          \
  } while (0)

// The value path. The stream already holds the reconstruction, so this is a
// conversion and no arithmetic.
#define QSQ_DEC_V(WT, IT, STORE)                                               \
  do {                                                                         \
    const IT *s = (const IT *)src;                                             \
    WT *d = (WT *)dst;                                                         \
    for (size_t i = 0; i < nbElts; ++i) {                                      \
      IT v = s[i];                                                             \
      v = v < flo ? flo : v;                                                   \
      d[i] = (WT)(STORE((double)v));                                           \
    }                                                                          \
  } while (0)

#define QSQ_ID(v) (v)
#define QSQ_TOF32(v) ((float)(v))
#define QSQ_TOF16(v) (quant_sqrt_float_to_half((float)(v)))

int quant_sqrt_decode(void *dst, const void *src, const quant_sqrt_params *p,
                      int dtype, size_t nbElts) {
  if (!QSQ_DTYPE_OK(dtype) || !isfinite(p->step) || !(p->step > 0.0) ||
      !isfinite(p->offset) || !(p->offset >= 0.0))
    return 1;
  const int values = (p->flags & QUANT_SQRT_FLAG_STORE_VALUES) != 0;
  const int nonneg = (p->flags & QUANT_SQRT_FLAG_NONNEGATIVE) != 0;
  const int narrow = (p->flags & QUANT_SQRT_FLAG_DECODE_F32) != 0;
  if (narrow && dtype != QSQ_F32)
    return 1;

  // An integer type carries the reconstruction in its own width, so the stream is
  // already the output.
  if (dtype <= QSQ_LAST_INT) {
    if (!values)
      return 1;
    memcpy(dst, src, nbElts * quant_sqrt_width(dtype));
    return 0;
  }
  if (nbElts == 0)
    return 0;

  if (values) {
    switch ((qsq_dtype)dtype) {
    case QSQ_F16: {
      const int16_t flo = nonneg ? 0 : INT16_MIN;
      QSQ_DEC_V(uint16_t, int16_t, QSQ_TOF16);
      break;
    }
    case QSQ_F32: {
      const int32_t flo = nonneg ? 0 : INT32_MIN;
      QSQ_DEC_V(float, int32_t, QSQ_TOF32);
      break;
    }
    default: {
      const int64_t flo = nonneg ? 0 : INT64_MIN;
      QSQ_DEC_V(double, int64_t, QSQ_ID);
      break;
    }
    }
    return 0;
  }

  const double step = p->step, offset = p->offset;
  const double vhi = quant_sqrt_value_hi(dtype);
  const double vlo = nonneg ? 0.0 : quant_sqrt_value_lo(dtype);
  const double dtop = quant_sqrt_index_top(step, offset, dtype);

  switch ((qsq_dtype)dtype) {
  case QSQ_F16: {
    const uint16_t utop = (uint16_t)(dtop > 32767.0 ? 32767.0 : dtop);
    QSQ_DEC(uint16_t, int16_t, uint16_t, float, QSQ_TOF16);
    break;
  }
  case QSQ_F32: {
    const uint32_t utop =
        (uint32_t)(dtop > 2147483647.0 ? 2147483647.0 : dtop);
    // The step was cut against whichever of these the resolver picked, so the bit
    // is not a hint, it says which arithmetic the declared bound was measured on.
    if (narrow)
      QSQ_DEC(float, int32_t, uint32_t, float, QSQ_ID);
    else
      QSQ_DEC(float, int32_t, uint32_t, double, QSQ_TOF32);
    break;
  }
  default: {
    const uint64_t utop =
        (uint64_t)(dtop > 9007199254740992.0 ? 9007199254740992.0 : dtop);
    QSQ_DEC(double, int64_t, uint64_t, double, QSQ_ID);
    break;
  }
  }
  return 0;
}
