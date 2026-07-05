#include "geozl/kernels.h"

#include "delta_w/auto_delta_w.h"
#include "delta_n/auto_delta_n.h"
#include "planar/auto_planar.h"
#include "med/auto_med.h"
#include "average/auto_average.h"
#include "wp_static/auto_wp_static.h"
#include "quant_linear/encode_quant_linear_kernel.h"
#include "quant_linear/decode_quant_linear_kernel.h"

// cffi calls the geozl_ names, each codec folder keeps its own. Renaming here
// keeps every folder self contained and copy-pasteable.

#define GEOZL_FWD_AUTO(codec)                                                 \
    size_t geozl_##codec##_encode_auto(                                       \
            void* dst, uint8_t* header, size_t header_cap, const void* src,   \
            size_t width, size_t nbElts, size_t eltWidth)                     \
    { return codec##_encode_auto(dst, header, header_cap, src, width,         \
                                 nbElts, eltWidth); }                         \
    void geozl_##codec##_decode_auto(                                         \
            void* dst, const void* src, const uint8_t* header,                \
            size_t headerSize, size_t nbElts, size_t eltWidth)                \
    { codec##_decode_auto(dst, src, header, headerSize, nbElts, eltWidth); }

GEOZL_FWD_AUTO(delta_w)
GEOZL_FWD_AUTO(delta_n)
GEOZL_FWD_AUTO(planar)
GEOZL_FWD_AUTO(med)
GEOZL_FWD_AUTO(average)
GEOZL_FWD_AUTO(wp_static)

void geozl_quant_linear_encode(void* dst, const void* src,
                               double scale, int dtype, size_t nbElts)
{ quant_linear_encode(dst, src, scale, dtype, nbElts); }

void geozl_quant_linear_decode(void* dst, const void* src,
                               double scale, int dtype, size_t nbElts)
{ quant_linear_decode(dst, src, scale, dtype, nbElts); }
