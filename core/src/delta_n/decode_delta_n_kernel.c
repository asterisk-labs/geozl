// Inverse vertical predictor. The dependency runs between rows, perpendicular
// to the row we vectorize, so each row is a plain element wise add with the
// reconstructed row above.

#include "decode_delta_n_kernel.h"

#include "common/raster.h" // geozl_row_width

#include <stdint.h>
#include <string.h>

#define DELTA_N_DEC(T)                                                         \
  do {                                                                         \
    T *d = (T *)dst;                                                           \
    const T *s = (const T *)src;                                               \
    memcpy(d, s, w * sizeof(T));                                               \
    for (size_t off = w; off < nbElts; off += w) {                             \
      const T *restrict up = d + off - w;                                      \
      const T *restrict res = s + off;                                         \
      T *restrict row = d + off;                                               \
      for (size_t c = 0; c < w; ++c)                                           \
        row[c] = (T)(res[c] + up[c]);                                          \
    }                                                                          \
  } while (0)

int delta_n_decode(void *dst, const void *src, size_t width, size_t nbElts,
                   size_t eltWidth) {
  // A width that does not divide nbElts would leave the last row short,
  // and the row loops below assume every row is complete.
  const size_t w = geozl_row_width(width, nbElts);
  if (w == 0)
    return 1;
  switch (eltWidth) {
  case 1:
    DELTA_N_DEC(uint8_t);
    break;
  case 2:
    DELTA_N_DEC(uint16_t);
    break;
  case 4:
    DELTA_N_DEC(uint32_t);
    break;
  case 8:
    DELTA_N_DEC(uint64_t);
    break;
  default:
    return 1; // eltWidth must be 1, 2, 4 or 8
  }
  return 0;
}

#undef DELTA_N_DEC
