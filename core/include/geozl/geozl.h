#ifndef GEOZL_H
#define GEOZL_H

#include "geozl/ctids.h"
#include "geozl/dtype.h"
#include "geozl/export.h"
#include "geozl/quant_linear_params.h"
#include "geozl/quant_log_params.h"
#include "geozl/quant_sqrt_params.h"

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
// The parameters are already resolved against the raster. Callers building a
// graph by hand get them from quant_linear_parse and quant_linear_resolve, or
// from geozl_lossy_parse and geozl_lossy_resolve for any of the three at once.
GEOZL_API ZL_NodeID geozl_node_quant_linear(ZL_Compressor *c,
                                            const quant_linear_params *params,
                                            int dtype);

// Parameters come from quant_log_parse and quant_log_resolve. The resolve reads a
// scan of the whole raster, which this node does not see, so it happens before
// the graph is built.
GEOZL_API ZL_NodeID geozl_node_quant_log(ZL_Compressor *c,
                                         const quant_log_params *params,
                                         int dtype);

// Parameters come from quant_sqrt_parse and quant_sqrt_resolve, and in the auto
// mode from quant_sqrt_fit before them. The node sees one flat stream, so none of
// that can happen inside it.
GEOZL_API ZL_NodeID geozl_node_quant_sqrt(ZL_Compressor *c,
                                          const quant_sqrt_params *params,
                                          int dtype);

// Missing-data modes, in the spirit of GDAL. NONE is a tile with nothing
// missing, NAN detects every non-finite sample itself and only applies to
// float, VALUE takes a sentinel the caller declares because a value like -9999
// is indistinguishable from a measurement without being told.
typedef enum {
  GEOZL_NODATA_NONE = 0,
  GEOZL_NODATA_NAN = 1,
  GEOZL_NODATA_VALUE = 2
} geozl_nodata_mode;

// mode is one of the three above. NONE has no node, since a tile with nothing
// missing wants no mask, so it is refused here rather than turned into an empty
// one. valueBits is the sentinel's bit pattern at the sample width, read only
// for VALUE and ignored for NAN.
GEOZL_API ZL_NodeID geozl_node_nodata(ZL_Compressor *c, uint32_t width,
                                      geozl_nodata_mode mode,
                                      uint64_t valueBits);

// 2d high-level compression through the graph method names, as geozl_2d_grid_c
// spells it, e.g. "planar>zigzag>transpose>entropy". The transpose and store_lo
// terminals need 2 to 8 bytes per element.
//
// error is a recipe too, so it crosses compress, bench and profile unchanged
// and the three cannot end up describing different errors:
//
//   NULL or ""                 lossless
//   "LINEAR:MAX_ERROR=V"       |x - x^| <= V
//   "LOG:MAX_ERROR=P%"         |x - x^| <= (P/100) * |x|
//   "SQRT:MAX_ERROR=VN"        |x - x^| <= V * sqrt(a + b*x)
//
// The recipe names the quantizer and the bound together, and geozl_lossy_parse
// reads the family off the prefix. Which one fits follows from how the
// measurement error of the data grows with the value, so a fixed instrument
// error takes LINEAR, photon counting takes SQRT, and a multiplicative error
// such as SAR speckle takes LOG. Each codec's spec.md carries the rest of its
// grammar.
//
// A SQRT recipe with no A and B fits the noise curve from this raster, since
// nothing here carries a product. Neighbouring tiles then land on different
// grids, so a caller that cuts a raster into tiles measures the curve once with
// quant_sqrt_accum and writes A and B into the recipe.

// nodataMode is a geozl_nodata_mode. nodataBits is the sentinel in the
// element's own representation, low eltWidth bytes, read only for
// GEOZL_NODATA_VALUE. Bits and not a double, which cannot carry an int64 or
// uint64 sentinel past 2^53.
//
// Returns 0 or the ZL_ErrorCode. The reason lands in errCtx, the size in
// *outSize.
GEOZL_API int geozl_2d_compress_c(const char *method, uint32_t width,
                                  const char *error, int dtype, int nodataMode,
                                  uint64_t nodataBits, const void *src,
                                  size_t numElts, size_t eltWidth, void *dst,
                                  size_t dstCapacity, size_t *outSize,
                                  char *errCtx, size_t errCtxSize);

// geozl_2d_compress_c split in two, for a caller with many tiles that share a
// graph. Same arguments up to the destination. src is read at open, since the
// error recipe is cut against it.
typedef struct geozl_2d_graph_s geozl_2d_graph;

GEOZL_API int geozl_2d_graph_open_c(geozl_2d_graph **out, const char *method,
                                    uint32_t width, const char *error,
                                    int dtype, int nodataMode,
                                    uint64_t nodataBits, const void *src,
                                    size_t numElts, size_t eltWidth,
                                    char *errCtx, size_t errCtxSize);

// The element width comes from the graph, numElts is this tile's.
GEOZL_API int geozl_2d_compress_graph_c(geozl_2d_graph *g, const void *src,
                                        size_t numElts, void *dst,
                                        size_t dstCapacity, size_t *outSize,
                                        char *errCtx, size_t errCtxSize);

GEOZL_API void geozl_2d_graph_close_c(geozl_2d_graph *g);

// Decompressed byte size of a frame, or 0 if it cannot be read.
GEOZL_API size_t geozl_2d_frame_dsize_c(const void *frame, size_t frameSize);

// Decompress a self-describing frame into dst. Returns 0 or the ZL_ErrorCode.
// verify == 0 skips both checksum verifications, worth 1 to 30 per cent of
// decode depending on how fast the frame already reads. It cannot add a
// checksum a frame does not carry, that is written at compression time.
GEOZL_API int geozl_2d_decompress_c(const void *frame, size_t frameSize,
                                    void *dst, size_t dstCapacity,
                                    size_t *outSize, int verify, char *errCtx,
                                    size_t errCtxSize);

// Times one graph, reps compressions and reps decompressions, all in C so the
// FFI is crossed once. Returns 0 or the ZL_ErrorCode of the first failing round
// trip. The nodata pair matches geozl_2d_compress_c, so the graph timed here is
// the one compress would build.
//
// checksum == 0 drops both checksums from the frame, so *compSize stops being
// the size compress writes; a lossy recipe drops the content one either way.
// verify is the decode side and costs no bytes. A profiler wants checksum 1 and
// verify 0.
GEOZL_API int geozl_2d_bench_c(const char *method, uint32_t width,
                               const char *error, int dtype, int nodataMode,
                               uint64_t nodataBits, const void *src,
                               size_t numElts, size_t eltWidth, size_t reps,
                               int checksum, int verify, size_t *compSize,
                               double *encSec, double *decSec, char *errCtx,
                               size_t errCtxSize);

// Recipe names of the grid a method expands to, one per stride-byte slot,
// count in *outCount. -1 on an unknown method.
GEOZL_API int geozl_2d_grid_c(const char *method, size_t eltWidth, char *names,
                              size_t stride, size_t maxNames, size_t *outCount);

#if defined(__cplusplus)
}
#endif

#endif // GEOZL_H