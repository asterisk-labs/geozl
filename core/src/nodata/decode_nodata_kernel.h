// Puts the missing samples back.
//
// The encoder replaced them with a fill, so the values stream carries a plain
// raster and the mask says which samples were never measured. Restoring is a
// straight write of the stored bit pattern wherever the mask says invalid.

#ifndef GEOZL_CODECS_NODATA_DECODE_KERNEL_H
#define GEOZL_CODECS_NODATA_DECODE_KERNEL_H

#include <stddef.h>
#include <stdint.h>

// Copies values to dst and writes @pattern wherever mask is 0. Follows GDAL,
// any nonzero byte counts as valid. dst may alias values.
void nodata_restore(void *dst, const void *values, const uint8_t *mask,
                    size_t nb_elts, size_t elt_width, uint64_t pattern);

// Writes @pattern into every sample, which is the whole of a tile where
// nothing was measured.
void nodata_broadcast(void *dst, size_t nb_elts, size_t elt_width,
                      uint64_t pattern);

#endif // GEOZL_CODECS_NODATA_DECODE_KERNEL_H
