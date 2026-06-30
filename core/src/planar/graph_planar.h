// Shared graph definition for the planar custom codec. Numeric in, numeric out.

#ifndef GEOZL_CODECS_PLANAR_GRAPH_H
#define GEOZL_CODECS_PLANAR_GRAPH_H

#include "openzl/zl_data.h" // ZL_Type_*, ZL_STREAMTYPELIST

// Codec id, 0x72D7xx block
#define PLANAR_CTID 0x72D703

// Local int param carrying the tile row width, set by the C graph builder
#define PLANAR_PARAM_WIDTH 1

#define PLANAR_GRAPH                                          \
    {                                                         \
        .CTid           = PLANAR_CTID,                        \
        .inStreamType   = ZL_Type_numeric,                   \
        .outStreamTypes = ZL_STREAMTYPELIST(ZL_Type_numeric) \
    }

#endif // GEOZL_CODECS_PLANAR_GRAPH_H
