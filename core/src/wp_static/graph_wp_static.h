// Shared graph definition for the wp_static custom codec. Numeric in, numeric
// out. wp_static freezes the weights of the JPEG XL weighted predictor into a
// signaled linear kernel, so the prediction is W plus a fixed combination of
// the row above. The kernel rides in the codec header, see spec.md.

#ifndef GEOZL_CODECS_WP_STATIC_GRAPH_H
#define GEOZL_CODECS_WP_STATIC_GRAPH_H

#include "openzl/zl_data.h"

// Codec id, 0x72D7xx block
#define WP_STATIC_CTID 0x72D707

// Causal stencil over the row above, in this order: N, NW, NE, NN.
#define WP_STATIC_NTAPS 4

// Local int params set by the graph builder. Width is the tile row width, the
// shift and the four coefficients are the trained kernel. Defaults are planar.
#define WP_STATIC_PARAM_WIDTH 1
#define WP_STATIC_PARAM_SHIFT 2
#define WP_STATIC_PARAM_C_N   3
#define WP_STATIC_PARAM_C_NW  4
#define WP_STATIC_PARAM_C_NE  5
#define WP_STATIC_PARAM_C_NN  6

#define WP_STATIC_GRAPH                                      \
    {                                                        \
        .CTid           = WP_STATIC_CTID,                    \
        .inStreamType   = ZL_Type_numeric,                   \
        .outStreamTypes = ZL_STREAMTYPELIST(ZL_Type_numeric) \
    }

#endif // GEOZL_CODECS_WP_STATIC_GRAPH_H
