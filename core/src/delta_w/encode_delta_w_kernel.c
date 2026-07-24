// Forward horizontal predictor. Offline path, plain scalar. Walk each row
// right to left so an in place transform keeps the source it still needs.

#include "encode_delta_w_kernel.h"

#include "common/raster.h" // geozl_row_width

#include <stdint.h>

#define DELTA_W_FWD(T)                                                         \
  do {                                                                         \
    T *d = (T *)dst;                                                           \
    const T *s = (const T *)src;                                               \
    for (size_t off = 0; off < nbElts; off += w) {                             \
      for (size_t c = w; c-- > 1;)                                             \
        d[off + c] = (T)(s[off + c] - s[off + c - 1]);                         \
      d[off] = s[off];                                                         \
    }                                                                          \
  } while (0)

int delta_w_encode(void *dst, const void *src, size_t width, size_t nbElts,
                   size_t eltWidth) {
  // A width that does not divide nbElts would leave the last row short,
  // and the row loops below assume every row is complete.
  const size_t w = geozl_row_width(width, nbElts);
  if (w == 0)
    return 1;
  switch (eltWidth) {
  case 1:
    DELTA_W_FWD(uint8_t);
    break;
  case 2:
    DELTA_W_FWD(uint16_t);
    break;
  case 4:
    DELTA_W_FWD(uint32_t);
    break;
  case 8:
    DELTA_W_FWD(uint64_t);
    break;
  default:
    return 1; // eltWidth must be 1, 2, 4 or 8
  }
  return 0;
}

#undef DELTA_W_FWD
