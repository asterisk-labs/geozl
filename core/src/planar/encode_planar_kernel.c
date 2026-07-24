// Forward planar predictor. Reads only from src and writes residuals to dst,
// so any traversal order is fine as long as dst does not alias src.

#include "encode_planar_kernel.h"

#include "common/raster.h" // geozl_row_width

#include <stdint.h>

#define PLANAR_FWD(T)                                                          \
  do {                                                                         \
    T *d = (T *)dst;                                                           \
    const T *s = (const T *)src;                                               \
    size_t rows = nbElts / w;                                                  \
    for (size_t r = 0; r < rows; ++r) {                                        \
      for (size_t c = 0; c < w; ++c) {                                         \
        size_t idx = r * w + c;                                                \
        T Wv = (c > 0) ? s[idx - 1] : 0;                                       \
        T Nv = (r > 0) ? s[idx - w] : 0;                                       \
        T NWv = (r > 0 && c > 0) ? s[idx - w - 1] : 0;                         \
        d[idx] = (T)(s[idx] - (T)(Wv + Nv - NWv));                             \
      }                                                                        \
    }                                                                          \
  } while (0)

int planar_encode(void *dst, const void *src, size_t width, size_t nbElts,
                  size_t eltWidth) {
  // A width that does not divide nbElts would leave the last row short,
  // and the row loops below assume every row is complete.
  const size_t w = geozl_row_width(width, nbElts);
  if (w == 0)
    return 1;
  switch (eltWidth) {
  case 1:
    PLANAR_FWD(uint8_t);
    break;
  case 2:
    PLANAR_FWD(uint16_t);
    break;
  case 4:
    PLANAR_FWD(uint32_t);
    break;
  case 8:
    PLANAR_FWD(uint64_t);
    break;
  default:
    return 1; // eltWidth must be 1, 2, 4 or 8
  }
  return 0;
}

#undef PLANAR_FWD
