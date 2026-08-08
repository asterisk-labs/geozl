// Puts the missing samples back, by writing the stored bit pattern wherever the
// mask says the sample was never measured.

#ifndef GEOZL_CODECS_NODATA_DECODE_KERNEL_H
#define GEOZL_CODECS_NODATA_DECODE_KERNEL_H

#include <stddef.h>
#include <stdint.h>

// Copies values to dst and writes @pattern wherever mask is 0. Follows GDAL,
// any nonzero byte counts as valid. dst may alias values.
void nodata_restore(void *dst, const void *values, const uint8_t *mask,
                    size_t nb_elts, size_t elt_width, uint64_t pattern);

#endif // GEOZL_CODECS_NODATA_DECODE_KERNEL_H
