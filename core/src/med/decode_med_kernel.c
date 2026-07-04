// Inverse MED predictor. The prediction clamps the gradient to the neighbour
// range, so it is nonlinear in W and cannot be a prefix sum the way a linear
// predictor is, each sample is reconstructed in turn from its decoded
// neighbours.

#include "decode_med_kernel.h"

#include <stdint.h>

#define MED_DEC(T)                                                          \
    do {                                                                     \
        T* d       = (T*)dst;                                                \
        const T* s = (const T*)src;                                          \
        size_t rows = nbElts / w;                                            \
        for (size_t r = 0; r < rows; ++r) {                                  \
            for (size_t c = 0; c < w; ++c) {                                 \
                size_t idx = r * w + c;                                      \
                T Wv  = (c > 0) ? d[idx - 1] : 0;                            \
                T Nv  = (r > 0) ? d[idx - w] : 0;                            \
                T NWv = (r > 0 && c > 0) ? d[idx - w - 1] : 0;               \
                T mn  = Wv < Nv ? Wv : Nv;                                   \
                T mx  = Wv < Nv ? Nv : Wv;                                   \
                T P   = (NWv >= mx) ? mn                                     \
                      : (NWv <= mn) ? mx                                     \
                                    : (T)(Wv + Nv - NWv);                    \
                d[idx] = (T)(s[idx] + P);                                    \
            }                                                                \
        }                                                                    \
    } while (0)

void med_decode(
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
        case 1: MED_DEC(uint8_t); break;
        case 2: MED_DEC(uint16_t); break;
        case 4: MED_DEC(uint32_t); break;
        case 8: MED_DEC(uint64_t); break;
        default: break;
    }
}

#undef MED_DEC
