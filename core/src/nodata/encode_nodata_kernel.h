// Splits a tile into values and a validity mask, then fills the holes.

#ifndef GEOZL_CODECS_NODATA_ENCODE_KERNEL_H
#define GEOZL_CODECS_NODATA_ENCODE_KERNEL_H

#include <stddef.h>
#include <stdint.h>

// Mask convention is GDAL's, 0 marks a sample to discard and 255 keeps it.
#define GEOZL_NODATA_INVALID 0
#define GEOZL_NODATA_VALID 255

// Bit pattern of the first NaN in the tile, IEEE only so elt_width is 2, 4 or
// 8. Returns 1 when one was found and 0 otherwise. Only NaN counts, an infinity
// is a value and travels as one.
int nodata_find_nan(uint64_t *pattern, const void *src, size_t nb_elts,
                    size_t elt_width);

// Marks every NaN, whatever its payload, and returns how many. A tile can carry
// more than one payload, and matching bits instead would leave all but the
// first kind for whatever runs next, which for quant_linear means a NaN the
// SPEC says it never has to handle. Widths other than 2, 4 and 8 hold no IEEE
// value, so the mask comes out all valid.
size_t nodata_mark_nan(uint8_t *mask, const void *src, size_t nb_elts,
                       size_t elt_width);

// Marks every sample whose bit pattern equals @pattern. Returns how many.
size_t nodata_mark_value(uint8_t *mask, const void *src, size_t nb_elts,
                         size_t elt_width, uint64_t pattern);

// Copies src to dst, replacing every marked sample. @width is the row width in
// samples, 0 or a width that does not divide nb_elts treats the tile as one
// row. dst may alias src.
void nodata_fill(void *dst, const void *src, const uint8_t *mask, size_t width,
                 size_t nb_elts, size_t elt_width);

#endif // GEOZL_CODECS_NODATA_ENCODE_KERNEL_H
