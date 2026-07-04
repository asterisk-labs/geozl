// Graph definition for the med codec, included by both bindings.

#ifndef GEOZL_CODECS_MED_GRAPH_H
#define GEOZL_CODECS_MED_GRAPH_H

#include "openzl/zl_data.h"

#define MED_CTID 0x72D705
#define MED_PARAM_WIDTH 1   // local int param key for the row width

#define MED_GRAPH                                            \
    {                                                        \
        .CTid           = MED_CTID,                          \
        .inStreamType   = ZL_Type_numeric,                   \
        .outStreamTypes = ZL_STREAMTYPELIST(ZL_Type_numeric) \
    }

#endif // GEOZL_CODECS_MED_GRAPH_H
