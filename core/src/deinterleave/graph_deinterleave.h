#ifndef GEOZL_CODECS_DEINTERLEAVE_GRAPH_H
#define GEOZL_CODECS_DEINTERLEAVE_GRAPH_H

#include "geozl/ctids.h"   // GEOZL_CTID_DEINTERLEAVE
#include "openzl/zl_data.h" // ZL_Type_numeric, ZL_STREAMTYPELIST

// One numeric stream in, two numeric streams out, the even and the odd lane.
// This is not the common predictor shape, so it carries its own graph.
// ZL_STREAMTYPELIST fills nbOutStreams, so it must be the last field.
#define GEOZL_DEINTERLEAVE_GRAPH                                                \
    {                                                                          \
        .CTid           = GEOZL_CTID_DEINTERLEAVE,                             \
        .inStreamType   = ZL_Type_numeric,                                     \
        .outStreamTypes = ZL_STREAMTYPELIST(ZL_Type_numeric, ZL_Type_numeric), \
    }

#endif // GEOZL_CODECS_DEINTERLEAVE_GRAPH_H
