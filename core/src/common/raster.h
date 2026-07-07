#ifndef GEOZL_COMMON_RASTER_H
#define GEOZL_COMMON_RASTER_H

#include <stddef.h>

// Effective row width for a predictor, or 0 when it cannot tile nbElts. On decode
// the width comes from the frame header, so it is not trusted. Call with nbElts != 0.
static inline size_t geozl_row_width(size_t width, size_t nbElts)
{
    const size_t w = (width == 0 || width > nbElts) ? nbElts : width;
    return (nbElts % w == 0) ? w : 0;
}

#endif // GEOZL_COMMON_RASTER_H