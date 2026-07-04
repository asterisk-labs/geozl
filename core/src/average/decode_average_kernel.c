// Inverse Average predictor. The prediction reads the reconstructed W, and
// enters it halved, so this is not a prefix sum like a unit weight predictor,
// each sample is reconstructed in turn from its decoded neighbours.

#include "decode_average_kernel.h"

#include <stdint.h>

#define AVERAGE_DEC(T)                                                      \
    do {                                                                     \
        T* d       = (T*)dst;                                                \
        const T* s = (const T*)src;                                          \
        size_t rows = nbElts / w;                                            \
        for (size_t r = 0; r < rows; ++r) {                                  \
            for (size_t c = 0; c < w; ++c) {                                 \
                size_t idx = r * w + c;                                      \
                T Wv = (c > 0) ? d[idx - 1] : 0;                             \
                T Nv = (r > 0) ? d[idx - w] : 0;                             \
                T P  = (T)((Wv >> 1) + (Nv >> 1) + (Wv & Nv & 1));           \
                d[idx] = (T)(s[idx] + P);                                    \
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
