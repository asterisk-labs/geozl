#ifndef GEOZL_CODECS_WP_STATIC_ENCODE_KERNEL_H
#define GEOZL_CODECS_WP_STATIC_ENCODE_KERNEL_H

#include <stddef.h>
#include <stdint.h>

// Forward wp_static predictor, the residual for wp_static_decode. @coeffs is
// {cN, cNW, cNE, cNN}. Row major, @width samples per row. @dst must not alias
// @src. See spec.md for the kernel and the edge convention.
void wp_static_encode(
        void* dst,
        const void* src,
        size_t width,
        size_t nbElts,
        size_t eltWidth,
        const int16_t coeffs[4],
        uint8_t shift);

#endif // GEOZL_CODECS_WP_STATIC_ENCODE_KERNEL_H
