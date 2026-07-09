#ifndef GEOZL_H
#define GEOZL_H

#include "geozl/ctids.h"
#include "geozl/export.h"

#include "openzl/zl_compressor.h" // ZL_Compressor, ZL_NodeID
#include "openzl/zl_ctransform.h" // ZL_NodeID_isValid
#include "openzl/zl_dtransform.h" // ZL_DCtx
#include "openzl/zl_errors.h"     // ZL_Report

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

// Decode. Register every geozl decoder into a DCtx before decompressing a frame
// that references a geozl codec. A frame built from OpenZL codecs alone needs
// none of this.
GEOZL_API ZL_Report geozl_register_decoders(ZL_DCtx* dctx);

// Check if a ctid is owned by geozl, and if it is lossy.
GEOZL_API int geozl_owns_ctid(uint32_t ctid);
GEOZL_API int geozl_ctid_is_lossy(uint32_t ctid);

// Encode. Each codec is a node builder. It registers the codec's typed encoder
// on the compressor, attaches the per tile parameters, and returns a node to
// chain into a graph. On failure it returns ZL_NODE_ILLEGAL, test the result
// with ZL_NodeID_isValid. width is the row width in samples.
GEOZL_API ZL_NodeID geozl_node_delta_w(ZL_Compressor* c, uint32_t width);
GEOZL_API ZL_NodeID geozl_node_delta_n(ZL_Compressor* c, uint32_t width);
GEOZL_API ZL_NodeID geozl_node_planar(ZL_Compressor* c, uint32_t width);
GEOZL_API ZL_NodeID geozl_node_med(ZL_Compressor* c, uint32_t width);
GEOZL_API ZL_NodeID geozl_node_average(ZL_Compressor* c, uint32_t width);
GEOZL_API ZL_NodeID geozl_node_wp_static(ZL_Compressor* c, uint32_t width);

// deinterleave splits a lane interleaved numeric stream into two by position, it
// takes no parameter. The lane semantics, complex components for instance, are
// the caller's, the codec only moves elements.
GEOZL_API ZL_NodeID geozl_node_deinterleave(ZL_Compressor* c);

// quant_linear is a lossy codec that quantizes a numeric stream to a error bound.
GEOZL_API ZL_NodeID geozl_node_quant_linear(ZL_Compressor* c, double max_error,
                                            int dtype);

#if defined(__cplusplus)
}
#endif

#endif // GEOZL_H