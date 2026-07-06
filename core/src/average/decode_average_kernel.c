// Inverse Average predictor, PNG filter 3 (RFC 2083 6.5,
// https://datatracker.ietf.org/doc/rfc2083/). Row zero and column zero 
// are peeled so the interior loop carries no boundary test.

#include "decode_average_kernel.h"

#include <stdint.h>

#define AVERAGE_DEC(T)                                                      \
    do {                                                                     \
        T* d              = (T*)dst;                                         \
        const T* s        = (const T*)src;                                   \
        const size_t rows = nbElts / w;                                      \
        d[0] = s[0];                                                         \
        for (size_t c = 1; c < w; ++c)                                       \
            d[c] = (T)(s[c] + (d[c - 1] >> 1));                             \
        for (size_t r = 1; r < rows; ++r) {                                  \
            const size_t row = r * w;                                        \
            d[row] = (T)(s[row] + (d[row - w] >> 1));                        \
            for (size_t c = 1; c < w; ++c) {                                 \
                const size_t i = row + c;                                    \
                const T Wv = d[i - 1];                                       \
                const T Nv = d[i - w];                                       \
                const T P  = (T)((Wv >> 1) + (Nv >> 1) + (Wv & Nv & 1));     \
                d[i] = (T)(s[i] + P);                                        \
            }                                                                \
        }                                                                    \
    } while (0)

void average_decode(
        void* dst,
        const void* src,
        size_t width,
        size_t nbElts,
        size_t eltWidth)
{
    if (nbElts == 0)
        return;
    const size_t w = (width == 0 || width > nbElts) ? nbElts : width;
    switch (eltWidth) {
        case 1: AVERAGE_DEC(uint8_t); break;
        case 2: AVERAGE_DEC(uint16_t); break;
        case 4: AVERAGE_DEC(uint32_t); break;
        case 8: AVERAGE_DEC(uint64_t); break;
        default: break;
    }
}

#undef AVERAGE_DEC