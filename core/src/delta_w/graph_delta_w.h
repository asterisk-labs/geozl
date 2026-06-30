// Shared graph definition for the delta_w custom codec, used by both the
// encoder and decoder registration. Numeric stream in, numeric stream out.

#ifndef GEOZL_CODECS_DELTA_W_GRAPH_H
#define GEOZL_CODECS_DELTA_W_GRAPH_H

#include "openzl/zl_data.h" // ZL_Type_*, ZL_STREAMTYPELIST

// Codec id, 0x72D7xx block
#define DELTA_W_CTID 0x72D701

// Local int param carrying the tile row width, set by the C graph builder
#define DELTA_W_PARAM_WIDTH 1

#define DELTA_W_GRAPH                                         \
    {                                                         \
        .CTid           = DELTA_W_CTID,                       \
        .inStreamType   = ZL_Type_numeric,                   \
        .outStreamTypes = ZL_STREAMTYPELIST(ZL_Type_numeric) \
    }

#endif // GEOZL_CODECS_DELTA_W_GRAPH_H
