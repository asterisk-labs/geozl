#include "encode_floatquant_kernel.h"

#include <stdint.h>

// to_latent_ordered: total order image of an IEEE float. Negative floats get
// all bits flipped, non negative floats get only the sign bit set, giving a
// monotonic unsigned key. The low mantissa bits are untouched for non negative
// values, which is why the split flips the residue for negatives so that an
// exactly quantized number always maps to secondary 0.
#define FLOATQUANT_SPLIT(T, SIGN)                                              \
  do {                                                                         \
    const T *s = (const T *)src;                                               \
    T *p = (T *)primary;                                                       \
    T *q = (T *)secondary;                                                     \
    const T maxlow = (T)(((T)1 << k) - (T)1);                                  \
    for (size_t i = 0; i < nb_elts; ++i) {                                     \
      const T u = s[i];                                                        \
      const T lat = (u & (SIGN)) ? (T)~u : (T)(u ^ (SIGN));                    \
      const T low = (T)(lat & maxlow);                                         \
      p[i] = (T)(lat >> k);                                                    \
      q[i] = (u & (SIGN)) ? (T)(maxlow - low) : low;                           \
    }                                                                          \
  } while (0)

void floatquant_split(void *primary, void *secondary, const void *src,
                      size_t nb_elts, size_t elt_width, unsigned k) {
  switch (elt_width) {
  case 4:
    FLOATQUANT_SPLIT(uint32_t, (uint32_t)0x80000000u);
    break;
  case 8:
    FLOATQUANT_SPLIT(uint64_t, (uint64_t)0x8000000000000000ull);
    break;
  default:
    break; // rejected by the binding
  }
}
