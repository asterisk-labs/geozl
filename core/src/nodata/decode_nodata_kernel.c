#include "decode_nodata_kernel.h"

#include <stdint.h>

// Branchless so a scattered mask costs the same as a coherent one, and so the
// loop still vectorizes. m is 0 or all ones, never a data dependent jump.
#define NODATA_RESTORE(T)                                                      \
  do {                                                                         \
    T *d = (T *)dst;                                                           \
    const T *v = (const T *)values;                                            \
    const T p = (T)pattern;                                                    \
    for (size_t i = 0; i < nb_elts; ++i) {                                     \
      const T m = (T)((T)0 - (T)(mask[i] == 0));                               \
      d[i] = (T)((v[i] & ~m) | (p & m));                                       \
    }                                                                          \
  } while (0)

void nodata_restore(void *dst, const void *values, const uint8_t *mask,
                    size_t nb_elts, size_t elt_width, uint64_t pattern) {
  switch (elt_width) {
  case 1:
    NODATA_RESTORE(uint8_t);
    break;
  case 2:
    NODATA_RESTORE(uint16_t);
    break;
  case 4:
    NODATA_RESTORE(uint32_t);
    break;
  case 8:
    NODATA_RESTORE(uint64_t);
    break;
  default:
    break; // rejected by the binding
  }
}
