// Shared graph definition for the delta_n custom codec. Numeric in, numeric out.

#ifndef GEOZL_CODECS_DELTA_N_GRAPH_H
#define GEOZL_CODECS_DELTA_N_GRAPH_H

#include "openzl/zl_data.h" // ZL_Type_*, ZL_STREAMTYPELIST

// Codec id, 0x72D7xx block
#define DELTA_N_CTID 0x72D702

// Local int param carrying the tile row width, set by the C graph builder
#define DELTA_N_PARAM_WIDTH 1

#define DELTA_N_GRAPH                                         \
    {                                                         \
        .CTid           = DELTA_N_CTID,                       \
        .inStreamType   = ZL_Type_numeric,                   \
        .outStreamTypes = ZL_STREAMTYPELIST(ZL_Type_numeric) \
    }

#endif // GEOZL_CODECS_DELTA_N_GRAPH_H
