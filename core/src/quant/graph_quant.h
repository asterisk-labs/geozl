// Shared graph definition for the quant lossy codec.
// Numeric in, numeric out. The output is always an integer index stream.

#ifndef GEOZL_CODECS_QUANT_GRAPH_H
#define GEOZL_CODECS_QUANT_GRAPH_H

#include "geozl/ctids.h"    // GEOZL_CTID_QUANT
#include "openzl/zl_data.h" // ZL_Type_*, ZL_STREAMTYPELIST

#include "quant_curve.h" // quant_params
#include "quant_dtype.h" // quant_dtype

#define QUANT_CTID GEOZL_CTID_QUANT

// Local params the graph builder sets on the encode node. dtype is an int
// param. The curve parameters ride in a copy param, they are already resolved
// against the tile by then and the int plane could not carry the doubles. The
// int and copy id planes are separate.
#define QUANT_PARAM_DTYPE 1
#define QUANT_PARAM_PARAMS 1

#define QUANT_GRAPH                                                            \
  {                                                                            \
    .CTid = QUANT_CTID, .inStreamType = ZL_Type_numeric,                       \
    .outStreamTypes = ZL_STREAMTYPELIST(ZL_Type_numeric)                       \
  }

// Wire header, little endian: uint8 dtype, uint8 curve, uint8 flags, then step
// and offset as IEEE doubles and nsub as a uint64.
#define QUANT_HEADER_SIZE (1 + 1 + 1 + 8 + 8 + 8)

#endif // GEOZL_CODECS_QUANT_GRAPH_H
