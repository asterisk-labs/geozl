// Forward wp_static predictor. Reads only from src, so any traversal order is
// fine as long as dst does not alias src. K is accumulated in the same width as
// the decoder, int32 for 8 and 16-bit samples, int64 for 32 and 64-bit, so both
// sides compute it identically.

#include "encode_wp_static_kernel.h"

#include <stddef.h>

#define WP_STATIC_FWD(T, ACC)                                                \
    do {                                                                      \
        T* d       = (T*)dst;                                                 \
        const T* s = (const T*)src;                                           \
        const size_t rows = nbElts / w;                                       \
        const ACC cN = coeffs[0], cNW = coeffs[1], cNE = coeffs[2],           \
                  cNN = coeffs[3];                                            \
        const ACC rnd = shift ? (ACC)1 << (shift - 1) : 0;                    \
        for (size_t r = 0; r < rows; ++r) {                                   \
            const size_t row = r * w;                                         \
            const T* ab  = (r >= 1) ? s + row - w : (const T*)0;              \
            const T* ab2 = (r >= 2) ? s + row - 2 * w : (const T*)0;          \
            for (size_t c = 0; c < w; ++c) {                                  \
                const ACC N  = ab ? (ACC)ab[c] : 0;                           \
                const ACC NW = (ab && c > 0) ? (ACC)ab[c - 1] : 0;            \
                const ACC NE = (ab && c + 1 < w) ? (ACC)ab[c + 1] : 0;        \
                const ACC NN = ab2 ? (ACC)ab2[c] : 0;                         \
                const ACC K  =                                                \
                    (cN * N + cNW * NW + cNE * NE + cNN * NN + rnd) >> shift;  \
                const T W = (c > 0) ? s[row + c - 1] : 0;                     \
                d[row + c] = (T)(s[row + c] - (T)((ACC)W + K));               \
            }                                                                 \
        }                                                                    \
    } while (0)

void wp_static_encode(
        void* dst,
        const void* src,
        size_t width,
        size_t nbElts,
        size_t eltWidth,
        const int16_t coeffs[4],
        uint8_t shift)
{
    if (nbElts == 0)
        return;
    const size_t w = (width == 0 || width > nbElts) ? nbElts : width;
    switch (eltWidth) {
        case 1: WP_STATIC_FWD(uint8_t,  int32_t); break;
        case 2: WP_STATIC_FWD(uint16_t, int32_t); break;
        case 4: WP_STATIC_FWD(uint32_t, int64_t); break;
        case 8: WP_STATIC_FWD(uint64_t, int64_t); break;
        default: break;
    }
}
