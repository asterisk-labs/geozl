#ifndef GEOZL_CODECS_QUANT_LINEAR_ENCODE_BINDING_H
#define GEOZL_CODECS_QUANT_LINEAR_ENCODE_BINDING_H

#include "graph_quant_linear.h"
#include "openzl/zl_ctransform.h" // ZL_Encoder

// C encode path. geozl encodes from Python today, this is the C side for a future
// C graph builder. Reads the dtype and scale from local int params, quantizes, and
// sends the 9 byte codec header.
ZL_Report EI_geozl_quant_linear(ZL_Encoder* eictx, const ZL_Input* ins[], size_t nbIns);

#endif // GEOZL_CODECS_QUANT_LINEAR_ENCODE_BINDING_H
