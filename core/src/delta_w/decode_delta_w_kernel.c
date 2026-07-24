// Inverse horizontal predictor, one prefix sum scan per row. The scan is hand
// written because a prefix sum does not auto vectorize. Native width modular
// wraparound, no zigzag, matching the geozl residual convention.

#include "decode_delta_w_kernel.h"

#include "common/raster.h" // geozl_row_width

#include <stdint.h>

#include "common/scan.h"

// The scans live in common/scan.h, shared by every horizontal predictor
// decoder. These aliases keep the local call sites unchanged.
#define scan8 geozl_scan8
#define scan16 geozl_scan16
#define scan32 geozl_scan32
#define scan64 geozl_scan64

#define DELTA_W_RUN(T, SCAN)                                                   \
  do {                                                                         \
    T *d = (T *)dst;                                                           \
    const T *s = (const T *)src;                                               \
    for (size_t off = 0; off < nbElts; off += w)                               \
      SCAN(d + off, s + off, w);                                               \
  } while (0)

int delta_w_decode(void *dst, const void *src, size_t width, size_t nbElts,
                   size_t eltWidth) {
  // A width that does not divide nbElts would leave the last row short,
  // and the row loops below assume every row is complete.
  const size_t w = geozl_row_width(width, nbElts);
  if (w == 0)
    return 1;
  switch (eltWidth) {
  case 1:
    DELTA_W_RUN(uint8_t, scan8);
    break;
  case 2:
    DELTA_W_RUN(uint16_t, scan16);
    break;
  case 4:
    DELTA_W_RUN(uint32_t, scan32);
    break;
  case 8:
    DELTA_W_RUN(uint64_t, scan64);
    break;
  default:
    return 1; // eltWidth must be 1, 2, 4 or 8
  }
  return 0;
}

#undef DELTA_W_RUN