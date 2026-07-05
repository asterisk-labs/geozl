#ifndef GEOZL_KERNELS_H
#define GEOZL_KERNELS_H

#include <stddef.h>
#include <stdint.h>

#include "geozl/export.h"

#ifdef __cplusplus
extern "C" {
#endif

// The cffi surface. Every lossless codec exposes the same pair. encode_auto
// estimates whatever parameters it needs, writes its own header, and encodes,
// returning the header size or 0 if header_cap is too small. decode_auto reads
// that header and decodes. The header layout lives in C, Python only moves the
// opaque bytes.
#define GEOZL_ENCODE_AUTO(codec)                                             \
    GEOZL_API size_t geozl_##codec##_encode_auto(                            \
            void* dst, uint8_t* header, size_t header_cap, const void* src,  \
            size_t width, size_t nb_elts, size_t elt_width)
#define GEOZL_DECODE_AUTO(codec)                                             \
    GEOZL_API void geozl_##codec##_decode_auto(                              \
            void* dst, const void* src, const uint8_t* header,               \
            size_t header_size, size_t nb_elts, size_t elt_width)

GEOZL_ENCODE_AUTO(delta_w);
GEOZL_DECODE_AUTO(delta_w);
GEOZL_ENCODE_AUTO(delta_n);
GEOZL_DECODE_AUTO(delta_n);
GEOZL_ENCODE_AUTO(planar);
GEOZL_DECODE_AUTO(planar);
GEOZL_ENCODE_AUTO(med);
GEOZL_DECODE_AUTO(med);
GEOZL_ENCODE_AUTO(average);
GEOZL_DECODE_AUTO(average);
GEOZL_ENCODE_AUTO(wp_static);
GEOZL_DECODE_AUTO(wp_static);

// quant_linear is lossy, its own contract, out of the auto family.
GEOZL_API void geozl_quant_linear_encode(void* dst, const void* src,
                                         double scale, int dtype, size_t nb_elts);
GEOZL_API void geozl_quant_linear_decode(void* dst, const void* src,
                                         double scale, int dtype, size_t nb_elts);

#ifdef __cplusplus
}
#endif

#endif // GEOZL_KERNELS_H
