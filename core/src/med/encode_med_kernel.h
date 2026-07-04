#ifndef GEOZL_CODECS_MED_ENCODE_KERNEL_H
#define GEOZL_CODECS_MED_ENCODE_KERNEL_H

#include <stddef.h>

// Forward MED predictor. @dst must not alias @src.
void med_encode(
        void* dst,
        const void* src,
        size_t width,
        size_t nbElts,
        size_t eltWidth);

#endif // GEOZL_CODECS_MED_ENCODE_KERNEL_H
