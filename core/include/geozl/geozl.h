#ifndef GEOZL_H
#define GEOZL_H

#include "geozl/ctids.h"
#include "geozl/export.h"

#include "openzl/zl_compressor.h"
#include "openzl/zl_ctransform.h"
#include "openzl/zl_dtransform.h"
#include "openzl/zl_errors.h"

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

GEOZL_API ZL_Report geozl_register_decoders(ZL_DCtx *dctx);

GEOZL_API int geozl_owns_ctid(uint32_t ctid);
GEOZL_API int geozl_ctid_is_lossy(uint32_t ctid);

// Node builders. Each registers the codec's encoder, attaches per-tile
// parameters, and returns a node to chain into a graph, or ZL_NODE_ILLEGAL on
// failure. width is the row width in samples.
GEOZL_API ZL_NodeID geozl_node_delta_w(ZL_Compressor *c, uint32_t width);
GEOZL_API ZL_NodeID geozl_node_delta_n(ZL_Compressor *c, uint32_t width);
GEOZL_API ZL_NodeID geozl_node_planar(ZL_Compressor *c, uint32_t width);
GEOZL_API ZL_NodeID geozl_node_med(ZL_Compressor *c, uint32_t width);
GEOZL_API ZL_NodeID geozl_node_average(ZL_Compressor *c, uint32_t width);
GEOZL_API ZL_NodeID geozl_node_wp_static(ZL_Compressor *c, uint32_t width);
GEOZL_API ZL_NodeID geozl_node_deinterleave(ZL_Compressor *c);
GEOZL_API ZL_NodeID geozl_node_binoffset(ZL_Compressor *c);
GEOZL_API ZL_NodeID geozl_node_intmult(ZL_Compressor *c, uint64_t base);
GEOZL_API ZL_NodeID geozl_node_floatquant(ZL_Compressor *c, unsigned k);
GEOZL_API ZL_NodeID geozl_node_floatmult(ZL_Compressor *c, double base);
GEOZL_API ZL_NodeID geozl_node_quant_linear(ZL_Compressor *c, double max_error,
                                            int dtype);

#if defined(__cplusplus)
}
#endif

#endif // GEOZL_H
