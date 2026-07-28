// Shared graph definition for the quant lossy codec.
// Numeric in, numeric out. The output is always an integer index stream.

#ifndef GEOZL_CODECS_QUANT_GRAPH_H
#define GEOZL_CODECS_QUANT_GRAPH_H

#include "geozl/ctids.h"    // GEOZL_CTID_QUANT
#include "openzl/zl_data.h" // ZL_Type_*, ZL_STREAMTYPELIST

#include "quant_curve.h" // quant_fwd, quant_inv
#include "quant_dtype.h" // quant_dtype
#include "quant_spec.h"  // quant_spec

#define QUANT_CTID GEOZL_CTID_QUANT

// Local params the graph builder sets on the encode node. Doubles cannot ride
// the int plane, so the parameters and the recipe go in copy params. Int and
// copy ids are separate spaces, hence DTYPE and PARAMS sharing a number. The
// recipe travels because the bound it declares is what the encoder measures
// itself against, and the resolved parameters no longer carry it.
#define QUANT_PARAM_DTYPE 1
#define QUANT_PARAM_PARAMS 1
#define QUANT_PARAM_SPEC 2

#define QUANT_GRAPH                                                            \
  {.CTid = QUANT_CTID,                                                         \
   .inStreamType = ZL_Type_numeric,                                            \
   .outStreamTypes = ZL_STREAMTYPELIST(ZL_Type_numeric)}

// Wire header, little endian. uint8 dtype, uint8 curve, uint8 flags, then step
// and offset as IEEE doubles and nsub as a uint64.
#define QUANT_HEADER_SIZE (1 + 1 + 1 + 8 + 8 + 8)

#endif // GEOZL_CODECS_QUANT_GRAPH_H