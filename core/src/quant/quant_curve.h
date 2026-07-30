// Curve arithmetic, shared by encode, decode and the spec resolver.

#ifndef GEOZL_CODECS_QUANT_CURVE_H
#define GEOZL_CODECS_QUANT_CURVE_H

#include "geozl/quant_params.h" // quant_params, quant_curve
#include "quant_dtype.h"        // quant_dtype
#include "quant_half.h"         // quant_float_to_half

#include <math.h>
#include <stdint.h>

// Spacing of the representable values near zero. Where b*|x| drops below it,
// nothing but x itself is inside the bound and no grid can serve that range.
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
// fixed fraction.
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

// Relative rounding of the output type, on top of the grid error.
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

// The grid ratio is exp(step) and rounding lands within exp(step/2), so the
// bound held is exp(step/2) - 1 and a bound of b needs step = 2*log1p(b).
static inline double quant_log_step(double b_rel) { return 2.0 * log1p(b_rel); }

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
    // offset is (nsub+1)*sub, so both directions recover sub the same way.
    if (p->nsub != 0 && a < p->offset) {
      m = nearbyint(a / (p->offset / ((double)p->nsub + 1.0)));
      if (m < 1.0)
        m = 1.0;
    } else {
      // a/offset overflows once the tile spans enough decades.
      m = (double)p->nsub + 1.0 +
          nearbyint((log(a) - log(p->offset)) / p->step);
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
    if (p->nsub != 0 && m <= (double)p->nsub) {
      a = m * (p->offset / ((double)p->nsub + 1.0));
    } else {
      const double t = (m - (double)p->nsub - 1.0) * p->step;
      a = p->offset * exp(t);
      // exp alone overflows well before the product does, since offset can
      // sit far below one. Halving the exponent keeps both in range.
      if (!isfinite(a) && isfinite(t)) {
        const double h = 0.5 * t;
        a = (p->offset * exp(h)) * exp(h);
      }
    }
    return q < 0.0 ? -a : a;
  }
  default:
    return q * p->step;
  }
}

// A warped reconstruction lands outside the output type at either end, the
// sqrt curve below zero and the log curve past the largest finite value.
// Clamping only moves it towards the data, so the bound holds. NaN goes to
// zero, since every comparison against it is false and a cast of it to an
// integer is undefined.
static inline double quant_clamp(double v, double lo, double hi) {
  if (v != v)
    return 0.0;
  return v < lo ? lo : (v > hi ? hi : v);
}

// The step arrives from a frame that may be damaged and the linear paths use it
// as an integer multiplier, so it saturates rather than casting out of range.
static inline uint64_t quant_step_u64(double step) {
  if (!(step > 1.0))
    return 1u; // also catches NaN
  if (step >= 18446744073709549568.0)
    return UINT64_MAX;
  return (uint64_t)step;
}

static inline int64_t quant_step_i64(double step) {
  if (!(step > 1.0))
    return 1;
  if (step >= 9223372036854774784.0)
    return INT64_MAX;
  return (int64_t)step;
}

// Range the index type can hold. Written out rather than taken from the limit
// macros, since (double)INT64_MAX rounds up to 2^63, which casts back to
// nothing. One copy, so no call site carries its own.
static inline double quant_index_lo(int dtype) {
  switch (dtype) {
  case Q_I8:
    return -128.0;
  case Q_I16:
  case Q_F16:
    return -32768.0;
  case Q_I32:
  case Q_F32:
    return -2147483648.0;
  case Q_I64:
  case Q_F64:
    return -9223372036854775808.0;
  default:
    return 0.0; // the unsigned widths
  }
}

static inline double quant_index_hi(int dtype) {
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
  case Q_F16:
    return 32767.0;
  case Q_I32:
  case Q_F32:
    return 2147483647.0;
  default:
    return 9223372036854774784.0;
  }
}

// Fold a computed index onto that range. A NaN has no index and lands on zero;
// a tile that means it should carry the nodata codec in front.
static inline double quant_index_fit(double q, double lo, double hi) {
  if (q != q)
    return 0.0;
  return q < lo ? lo : (q > hi ? hi : q);
}

// Range of the output type. Same reason as the index range for writing the
// integer limits out, and encode and decode both read them from here so a
// reconstruction cannot be in range on one side and wrapped on the other.
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
    return -3.40282346638528860e38;
  case Q_F64:
    return -1.79769313486231571e308;
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
  case Q_F32:
    return 3.40282346638528860e38;
  default:
    return 1.79769313486231571e308;
  }
}

// Floor of the reconstruction. The type minimum, unless the encoder measured
// that the tile held nothing negative, in which case zero. Clamping toward an
// interval that contains x can only shorten |x - x^|, so the bound survives
// either way and this only ever moves a reconstruction closer to its sample.
static inline double quant_floor(const quant_params *p, int dtype) {
  if ((p->flags & QUANT_FLAG_NONNEGATIVE) != 0)
    return 0.0;
  return quant_value_lo(dtype);
}


// How a reconstruction reaches the output width, on both sides, so the index
// encode picks as nearest is the one that comes back. The rounding is spelled
// out because a cast to an integer truncates.
#define QUANT_RT_INT(v, lo, hi) nearbyint(quant_clamp((v), (lo), (hi)))
#define QUANT_RT_F16(v, lo, hi)                                                \
  ((double)quant_half_to_float(                                                \
      quant_float_to_half((float)quant_clamp((v), (lo), (hi)))))
#define QUANT_RT_F32(v, lo, hi) ((double)(float)quant_clamp((v), (lo), (hi)))
#define QUANT_RT_F64(v, lo, hi) quant_clamp((v), (lo), (hi))

#endif // GEOZL_CODECS_QUANT_CURVE_H