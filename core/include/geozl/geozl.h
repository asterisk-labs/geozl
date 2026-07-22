#ifndef GEOZL_H
#define GEOZL_H

#include "geozl/ctids.h"
#include "geozl/export.h"

#include "openzl/zl_compressor.h"
#include "openzl/zl_ctransform.h"
#include "openzl/zl_dtransform.h"
#include "openzl/zl_errors.h"

#include <stddef.h>
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

// 2d high-level compression. A tile is compressed through one graph chosen out
// of the grid {predictor prior + id} x {entropy, field_lz, zstd, store_lo}.
// method is the prior (a name, "none", or NULL for unbiased). graph runs one
// candidate directly by recipe name. optim picks what the selector minimizes
// (store/speed/balanced), lam is the balanced weight. max_error < 0 is lossless.
#define GEOZL_2D_LOSSLESS (-1.0)

#define GEOZL_2D_OPTIM_STORE 0
#define GEOZL_2D_OPTIM_SPEED 1
#define GEOZL_2D_OPTIM_BALANCED 2

GEOZL_API ZL_Report geozl_2d_compress(const char *method, const char *graph,
                                      int optim, double lam, uint32_t width,
                                      double max_error, int dtype,
                                      const void *src, size_t numElts,
                                      size_t eltWidth, void *dst,
                                      size_t dstCapacity, size_t *outSize,
                                      char *chosenGraph, size_t chosenGraphSize,
                                      char *errCtx, size_t errCtxSize);

// Returns 0 or the ZL_ErrorCode. The winning recipe lands in chosenGraph
// (may be NULL), the reason in errCtx, the size in *outSize.
GEOZL_API int geozl_2d_compress_c(const char *method, const char *graph,
                                  int optim, double lam, uint32_t width,
                                  double max_error, int dtype, const void *src,
                                  size_t numElts, size_t eltWidth, void *dst,
                                  size_t dstCapacity, size_t *outSize,
                                  char *chosenGraph, size_t chosenGraphSize,
                                  char *errCtx, size_t errCtxSize);

// Decompressed byte size of a frame, or 0 if it cannot be read.
GEOZL_API size_t geozl_2d_frame_dsize_c(const void *frame, size_t frameSize);

// Decompress a self-describing frame into dst. Returns 0 or the ZL_ErrorCode.
GEOZL_API int geozl_2d_decompress_c(const void *frame, size_t frameSize,
                                    void *dst, size_t dstCapacity,
                                    size_t *outSize, char *errCtx,
                                    size_t errCtxSize);

// Recipe names of the grid a method expands to, one per stride-byte slot,
// count in *outCount. -1 on an unknown method.
GEOZL_API int geozl_2d_grid_c(const char *method, size_t eltWidth, char *names,
                              size_t stride, size_t maxNames, size_t *outCount);

#if defined(__cplusplus)
}
#endif

#endif // GEOZL_H