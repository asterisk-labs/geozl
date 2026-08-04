#ifndef GEOZL_CODECS_QUANT_LINEAR_ENCODE_KERNEL_H
#define GEOZL_CODECS_QUANT_LINEAR_ENCODE_KERNEL_H

#include "geozl/quant_linear_params.h"

#include <stddef.h>

// Quantize onto a uniform grid anchored at zero. The only lossy step. The stream
// keeps the element width dtype names, so dst must hold nbElts of it, and carries
// the index unless QUANT_LINEAR_FLAG_STORE_VALUES asks for the reconstruction.
// Returns 0, or nonzero for a dtype or step the resolver cannot have produced.
int quant_linear_encode(void *restrict dst, const void *restrict src,
                        const quant_linear_params *p, int dtype, size_t nbElts);

// Largest magnitude, and whether anything was negative. Non-finite samples are
// skipped: they have no index and the nodata codec in front of a lossy graph is
// what puts them back. A finite sentinel is not skipped, because nothing here tells
// one from a reading.
// Returns 0 when the tile holds a nonzero finite sample, 1 otherwise.
int quant_linear_scan(const void *src, int dtype, size_t nbElts,
                      quant_linear_stats *out);

#endif // GEOZL_CODECS_QUANT_LINEAR_ENCODE_KERNEL_H
