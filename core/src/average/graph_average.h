// Graph definition for the average codec, included by both bindings.

#ifndef GEOZL_CODECS_AVERAGE_GRAPH_H
#define GEOZL_CODECS_AVERAGE_GRAPH_H

#include "openzl/zl_data.h"

#define AVERAGE_CTID 0x72D706
#define AVERAGE_PARAM_WIDTH 1   // local int param key for the row width

#define AVERAGE_GRAPH                                        \
    {                                                        \
        .CTid           = AVERAGE_CTID,                      \
        .inStreamType   = ZL_Type_numeric,                   \
        .outStreamTypes = ZL_STREAMTYPELIST(ZL_Type_numeric) \
    }

#endif // GEOZL_CODECS_AVERAGE_GRAPH_H
