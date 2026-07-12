#ifndef GEOZL_CODECS_AVERAGE_ENCODE_KERNEL_H
#define GEOZL_CODECS_AVERAGE_ENCODE_KERNEL_H

#include <stddef.h>

// Forward Average predictor. @dst must not alias @src.
void average_encode(void *dst, const void *src, size_t width, size_t nbElts,
                    size_t eltWidth);

#endif // GEOZL_CODECS_AVERAGE_ENCODE_KERNEL_H
