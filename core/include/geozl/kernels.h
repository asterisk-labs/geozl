#ifndef GEOZL_KERNELS_H
#define GEOZL_KERNELS_H

#include <stddef.h>
#include "geozl/export.h"

#ifdef __cplusplus
extern "C" {
#endif

// Pure C, no OpenZL. The Python side calls these directly over cffi.
GEOZL_API void geozl_delta_w_encode(void* dst, const void* src,
                                    size_t width, size_t nb_elts, size_t elt_width);
GEOZL_API void geozl_delta_w_decode(void* dst, const void* src,
                                    size_t width, size_t nb_elts, size_t elt_width);
GEOZL_API void geozl_delta_n_encode(void* dst, const void* src,
                                    size_t width, size_t nb_elts, size_t elt_width);
GEOZL_API void geozl_delta_n_decode(void* dst, const void* src,
                                    size_t width, size_t nb_elts, size_t elt_width);
GEOZL_API void geozl_planar_encode(void* dst, const void* src,
                                   size_t width, size_t nb_elts, size_t elt_width);
GEOZL_API void geozl_planar_decode(void* dst, const void* src,
                                   size_t width, size_t nb_elts, size_t elt_width);
GEOZL_API void geozl_med_encode(void* dst, const void* src,
                                size_t width, size_t nb_elts, size_t elt_width);
GEOZL_API void geozl_med_decode(void* dst, const void* src,
                                size_t width, size_t nb_elts, size_t elt_width);
GEOZL_API void geozl_average_encode(void* dst, const void* src,
                                    size_t width, size_t nb_elts, size_t elt_width);
GEOZL_API void geozl_average_decode(void* dst, const void* src,
                                    size_t width, size_t nb_elts, size_t elt_width);
GEOZL_API void geozl_quant_linear_encode(void* dst, const void* src,
                                         double scale, int dtype, size_t nb_elts);
GEOZL_API void geozl_quant_linear_decode(void* dst, const void* src,
                                         double scale, int dtype, size_t nb_elts);

#ifdef __cplusplus
}
#endif

#endif