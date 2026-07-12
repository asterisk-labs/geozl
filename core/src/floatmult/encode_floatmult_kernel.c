#include "encode_floatmult_kernel.h"
#include "floatmult_common.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

// round() matches Rust f64::round (half away from zero). The adjustment is the
// full latent gap, so the round trip is exact for any base, the base only moves
// the compression ratio. base and inv_base are passed as f64 and narrowed to
// f32 inside the width-4 path so encode and decode share the same arithmetic.

void floatmult_split(void *primary, void *secondary, const void *src,
                     size_t nb_elts, size_t elt_width, double base,
                     double inv_base) {
  if (elt_width == 4) {
    const float b = (float)base;
    const float invb = (float)inv_base;
    const uint32_t *s = (const uint32_t *)src;
    uint32_t *p = (uint32_t *)primary;
    uint32_t *q = (uint32_t *)secondary;
    for (size_t i = 0; i < nb_elts; ++i) {
      float x;
      memcpy(&x, &s[i], 4);
      float mult = roundf(x * invb);
      // non finite x (or overflow) has no grid multiple; pin mult to 0 so
      // the adjustment carries the whole latent gap and x round trips
      if (!isfinite(mult))
        mult = 0.0f;
      const float prod = mult * b;
      uint32_t xb, pb;
      memcpy(&xb, &x, 4);
      memcpy(&pb, &prod, 4);
      p[i] = fm_int_to_latent32(mult);
      q[i] = (uint32_t)(fm_ord32(xb) - fm_ord32(pb)) + 0x80000000u; // toggle
    }
  } else if (elt_width == 8) {
    const double b = base;
    const double invb = inv_base;
    const uint64_t *s = (const uint64_t *)src;
    uint64_t *p = (uint64_t *)primary;
    uint64_t *q = (uint64_t *)secondary;
    for (size_t i = 0; i < nb_elts; ++i) {
      double x;
      memcpy(&x, &s[i], 8);
      double mult = round(x * invb);
      if (!isfinite(mult))
        mult = 0.0;
      const double prod = mult * b;
      uint64_t xb, pb;
      memcpy(&xb, &x, 8);
      memcpy(&pb, &prod, 8);
      p[i] = fm_int_to_latent64(mult);
      q[i] = (uint64_t)(fm_ord64(xb) - fm_ord64(pb)) +
             0x8000000000000000ull; // toggle
    }
  }
  // other widths rejected by the binding
}
