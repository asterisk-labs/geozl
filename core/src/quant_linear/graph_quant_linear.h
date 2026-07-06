// Shared graph definition for the quant_linear lossy codec.
// Numeric in, numeric out. The output is always an integer index stream.

#ifndef GEOZL_CODECS_QUANT_LINEAR_GRAPH_H
#define GEOZL_CODECS_QUANT_LINEAR_GRAPH_H

#include "geozl/ctids.h"    // GEOZL_CTID_QUANT_LINEAR
#include "openzl/zl_data.h" // ZL_Type_*, ZL_STREAMTYPELIST

#define QUANT_LINEAR_CTID GEOZL_CTID_QUANT_LINEAR

#include "quant_linear_dtype.h" // ql_dtype

// Local params the graph builder sets on the encode node. dtype is an int
// param. scale is a double, so it rides in a copy param, eight bytes, never an
// int param which would truncate it. The int and copy id planes are separate.
#define QUANT_LINEAR_PARAM_DTYPE 1
#define QUANT_LINEAR_PARAM_SCALE 1

#define QUANT_LINEAR_GRAPH                                    \
    {                                                         \
        .CTid           = QUANT_LINEAR_CTID,                  \
        .inStreamType   = ZL_Type_numeric,                    \
        .outStreamTypes = ZL_STREAMTYPELIST(ZL_Type_numeric)  \
    }

#endif // GEOZL_CODECS_QUANT_LINEAR_GRAPH_H
