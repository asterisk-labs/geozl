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

#endif // GEOZL_COMMON_RASTER_H