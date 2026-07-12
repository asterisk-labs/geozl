#ifndef GEOZL_CODECS_AVERAGE_DECODE_KERNEL_H
#define GEOZL_CODECS_AVERAGE_DECODE_KERNEL_H

#include <stddef.h>

// Inverse Average predictor, row major, @width samples per row. @dst may alias
// @src.
void average_decode(void *dst, const void *src, size_t width, size_t nbElts,
                    size_t eltWidth);

#endif // GEOZL_CODECS_AVERAGE_DECODE_KERNEL_H
