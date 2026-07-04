#ifndef GEOZL_CODECS_WP_STATIC_DECODE_KERNEL_H
#define GEOZL_CODECS_WP_STATIC_DECODE_KERNEL_H

#include <stddef.h>
#include <stdint.h>

// Inverse wp_static predictor, W plus a signaled linear kernel over the row
// above. @coeffs is {cN, cNW, cNE, cNN}. Row major, @width samples per row.
// @dst may alias @src. See spec.md for the kernel and the edge convention.
void wp_static_decode(
        void* dst,
        const void* src,
        size_t width,
        size_t nbElts,
        size_t eltWidth,
        const int16_t coeffs[4],
        uint8_t shift);

#endif // GEOZL_CODECS_WP_STATIC_DECODE_KERNEL_H
