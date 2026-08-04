#ifndef GEOZL_COMMON_RASTER_H
#define GEOZL_COMMON_RASTER_H

#include <stddef.h>

// Effective row width for a predictor, or 0 when it cannot tile nbElts. Also 0
// for nbElts == 0, so a single check covers both the empty input and a width
// that does not divide. On decode the width comes from the frame header, so it
// is not trusted.
static inline size_t geozl_row_width(size_t width, size_t nbElts) {
  if (nbElts == 0)
    return 0;
  const size_t w = (width == 0 || width > nbElts) ? nbElts : width;
  return (nbElts % w == 0) ? w : 0;
}

// Row width plus the element width switch, written out identically by nine
// kernels. B is the width in bits, for bodies that paste it into a scan name.
// dst and src must be aligned to eltWidth; checking it here costs delta_w u64
// a third, so it belongs where an unaligned pointer first appears.
#define GEOZL_ROW_DISPATCH(W, BODY)                                            \
  do {                                                                         \
    const size_t W = geozl_row_width(width, nbElts);                           \
    if (W == 0)                                                                \
      return 1;                                                                \
    switch (eltWidth) {                                                        \
    case 1:                                                                    \
      BODY(uint8_t, 8);                                                        \
      break;                                                                   \
    case 2:                                                                    \
      BODY(uint16_t, 16);                                                      \
      break;                                                                   \
    case 4:                                                                    \
      BODY(uint32_t, 32);                                                      \
      break;                                                                   \
    case 8:                                                                    \
      BODY(uint64_t, 64);                                                      \
      break;                                                                   \
    default:                                                                   \
      return 1; /* an OpenZL numeric stream is 1, 2, 4 or 8 */                 \
    }                                                                          \
    return 0;                                                                  \
  } while (0)

#endif // GEOZL_COMMON_RASTER_H