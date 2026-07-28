// Curve arithmetic, shared by encode, decode and the spec resolver.

#ifndef GEOZL_CODECS_QUANT_CURVE_H
#define GEOZL_CODECS_QUANT_CURVE_H

#include "geozl/quant_params.h" // quant_params, quant_curve
#include "quant_dtype.h"        // quant_dtype
#include "quant_half.h"         // quant_float_to_half

#include <math.h>
#include <stdint.h>

// Spacing of the representable values near zero, one for integers and the
// smallest subnormal for floats. Where b*|x| drops below that spacing, nothing
// but x itself is inside the bound, so the log curve carries that range as its
// leading nsub indices rather than quantizing it.
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

// Where the neighbours stop being a fixed distance apart and start being a
// fixed fraction. The smallest normal for floats, one for integers.
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

// The log grid has ratio exp(step) and rounding to the nearest level lands
// within a factor exp(step/2), so the bound it holds is exp(step/2) - 1. The
// other way round, a bound of b needs step = 2*log1p(b).
static inline double quant_log_step(double b_rel) { return 2.0 * log1p(b_rel); }

static inline double quant_log_brel(double step) { return expm1(0.5 * step); }

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
    // offset is (nsub+1)*sub, so sub comes back out of the header instead of
    // the dtype and both directions divide by the same double.
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

// A warped reconstruction can land outside the output type, the sqrt curve
// below zero in particular, which would wrap on an unsigned width. Clamping
// only moves it towards the data, so the bound holds.
//
// The limits sit here so encode and decode read the same ones. The integer
// limits are the largest of each width that survive a round trip through a
// double.
static inline double quant_clamp(double v, double lo, double hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

static inline double quant_value_lo(int dtype) {
  switch (dtype) {
  case Q_I8:
    return -128.0;
  case Q_I16:
    return -32768.0;
  case Q_I32:
    return -2147483648.0;
  case Q_I64:
    return -9223372036854775808.0;
  case Q_F16:
    return -65504.0;
  case Q_F32:
  case Q_F64:
    return -INFINITY;
  default:
    return 0.0; // the unsigned widths
  }
}

static inline double quant_value_hi(int dtype) {
  switch (dtype) {
  case Q_U8:
    return 255.0;
  case Q_U16:
    return 65535.0;
  case Q_U32:
    return 4294967295.0;
  case Q_U64:
    return 18446744073709549568.0;
  case Q_I8:
    return 127.0;
  case Q_I16:
    return 32767.0;
  case Q_I32:
    return 2147483647.0;
  case Q_I64:
    return 9223372036854774784.0;
  case Q_F16:
    return 65504.0;
  default:
    return INFINITY;
  }
}

// How a reconstruction reaches the output width. Encode and decode both go
// through these, so the index encode picks as nearest is the one that comes
// back. The rounding is spelled out because a cast to an integer truncates,
// which would leave every warped reconstruction up to a unit off.
#define QUANT_RT_INT(v, lo, hi) nearbyint(quant_clamp((v), (lo), (hi)))
#define QUANT_RT_F16(v, lo, hi)                                                \
  ((double)quant_half_to_float(                                                \
      quant_float_to_half((float)quant_clamp((v), (lo), (hi)))))
#define QUANT_RT_F32(v, lo, hi) ((double)(float)quant_clamp((v), (lo), (hi)))
#define QUANT_RT_F64(v, lo, hi) quant_clamp((v), (lo), (hi))

#endif // GEOZL_CODECS_QUANT_CURVE_H