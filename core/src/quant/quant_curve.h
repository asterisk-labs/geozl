// Arithmetic of the three curves declared in geozl/quant_params.h, shared by
// encode, decode and the spec resolver so one definition serves all three and
// they cannot drift.

#ifndef GEOZL_CODECS_QUANT_CURVE_H
#define GEOZL_CODECS_QUANT_CURVE_H

#include "geozl/quant_params.h" // quant_params, quant_curve
#include "quant_dtype.h"          // quant_dtype

#include <math.h>
#include <stdint.h>

// Spacing of the representable values of dtype near zero: one for integers,
// the smallest positive subnormal for floats. Where that spacing is coarse
// against the bound, no representable value other than x itself lies within
// b*|x| of x, so no grid can serve that range and the log curve stores it
// exactly instead, as nsub leading index values. That is what the second
// clause of a relative bound ("or the two are identical") is for, and it is
// the case that defeats rounding the mantissa to a fixed number of bits.
static inline double quant_sub(int dtype) {
  switch (dtype) {
  case Q_F16:
    return 5.9604644775390625e-8; // 2^-24
  case Q_F32:
    return 1.40129846432481707e-45; // 2^-149
  case Q_F64:
    return 4.9406564584124654e-324; // 2^-1074
  default:
    return 1.0;
  }
}

// Smallest value of dtype whose neighbours are a fixed fraction away rather
// than a fixed distance: the smallest normal for floats, and one for integers,
// below which nothing is representable at all.
static inline double quant_normal_min(int dtype) {
  switch (dtype) {
  case Q_F16:
    return 6.103515625e-5; // 2^-14
  case Q_F32:
    return 1.17549435082228751e-38; // 2^-126
  case Q_F64:
    return 2.22507385850720138e-308; // 2^-1022
  default:
    return 1.0;
  }
}

// Relative rounding of the output type, added on top of the grid error when the
// reconstruction is stored back at that width.
static inline double quant_eps(int dtype) {
  switch (dtype) {
  case Q_F16:
    return 4.8828125e-4; // 2^-11
  case Q_F32:
    return 5.9604644775390625e-8; // 2^-24
  case Q_F64:
    return 1.1102230246251565e-16; // 2^-53
  default:
    return 0.0;
  }
}

// Grid ratio of the log curve is exp(step), and rounding to the nearest level
// lands within a factor exp(step/2), so the relative bound it holds is
// exp(step/2) - 1. Inverted, a bound of b needs step = 2*log1p(b).
static inline double quant_log_step(double b_rel) { return 2.0 * log1p(b_rel); }

static inline double quant_log_brel(double step) {
  return expm1(0.5 * step);
}

// x -> index. Returns a double, the kernels clamp and store at the index width.
static inline double quant_fwd(double x, const quant_params *p) {
  if (p->step == 0.0)
    return x;
  switch (p->curve) {
  case QUANT_CURVE_SQRT: {
    const double t = x + p->offset;
    return nearbyint(sqrt(t < 0.0 ? 0.0 : t) / p->step);
  }
  case QUANT_CURVE_LOG: {
    const double a = fabs(x);
    if (a == 0.0)
      return 0.0;
    double m;
    // offset is (nsub+1)*sub by construction, so the exact grid spacing comes
    // back out of the header rather than from the dtype, and the two directions
    // divide by the same double.
    if (p->nsub != 0 && a < p->offset) {
      m = nearbyint(a / (p->offset / ((double)p->nsub + 1.0)));
      if (m < 1.0)
        m = 1.0;
    } else {
      m = (double)p->nsub + 1.0 + nearbyint(log(a / p->offset) / p->step);
      if (m < (double)p->nsub + 1.0)
        m = (double)p->nsub + 1.0;
    }
    return x < 0.0 ? -m : m;
  }
  default:
    return nearbyint(x / p->step);
  }
}

// index -> x.
static inline double quant_inv(double q, const quant_params *p) {
  if (p->step == 0.0)
    return q;
  switch (p->curve) {
  case QUANT_CURVE_SQRT: {
    const double w = q * p->step;
    return w * w - p->offset;
  }
  case QUANT_CURVE_LOG: {
    const double m = fabs(q);
    if (m == 0.0)
      return 0.0;
    double a;
    if (p->nsub != 0 && m <= (double)p->nsub)
      a = m * (p->offset / ((double)p->nsub + 1.0));
    else
      a = p->offset * exp((m - (double)p->nsub - 1.0) * p->step);
    return q < 0.0 ? -a : a;
  }
  default:
    return q * p->step;
  }
}

// Pointwise bound the parameters hold at x, before the reconstruction is
// rounded back to the output width. The encoder verifies against it rather
// than trusting the algebra, so libm accuracy cannot turn into a frame that
// exceeds its declared error.
static inline double quant_bound(double x, const quant_params *p) {
  if (p->step == 0.0)
    return 0.0;
  switch (p->curve) {
  case QUANT_CURVE_SQRT: {
    const double t = x + p->offset;
    return p->step * sqrt(t < 0.0 ? 0.0 : t);
  }
  case QUANT_CURVE_LOG:
    return quant_log_brel(p->step) * fabs(x);
  default:
    return 0.5 * p->step;
  }
}

#endif // GEOZL_CODECS_QUANT_CURVE_H
