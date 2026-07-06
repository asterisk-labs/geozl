// Forward Average predictor (PNG filter 3). The floor average of W and N is
// computed as (W>>1) + (N>>1) + (W&N&1) so the sum never overflows the sample
// width.

#include "encode_average_kernel.h"

#include <stdint.h>

#define AVERAGE_FWD(T)                                                      \
    do {                                                                     \
        T* d       = (T*)dst;                                                \
        const T* s = (const T*)src;                                          \
        size_t rows = nbElts / w;                                            \
        for (size_t r = 0; r < rows; ++r) {                                  \
            for (size_t c = 0; c < w; ++c) {                                 \
                size_t idx = r * w + c;                                      \
                T Wv = (c > 0) ? s[idx - 1] : 0;                             \
                T Nv = (r > 0) ? s[idx - w] : 0;                             \
                T P  = (T)((Wv >> 1) + (Nv >> 1) + (Wv & Nv & 1));           \
                d[idx] = (T)(s[idx] - P);                                    \
            }                                                                \
        }                                                                    \
    } while (0)

void average_encode(
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
        case 1: AVERAGE_FWD(uint8_t); break;
        case 2: AVERAGE_FWD(uint16_t); break;
        case 4: AVERAGE_FWD(uint32_t); break;
        case 8: AVERAGE_FWD(uint64_t); break;
        default: break;
    }
}

#undef AVERAGE_FWD
