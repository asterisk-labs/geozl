#ifndef GEOZL_CODECS_QUANT_SQRT_ENCODE_KERNEL_H
#define GEOZL_CODECS_QUANT_SQRT_ENCODE_KERNEL_H

#include "geozl/quant_sqrt_params.h"

#include <stddef.h>

// Samples into the stream. dst is a separate buffer of nbElts elements at the
// width of dtype, and carries the index unless QUANT_SQRT_FLAG_STORE_VALUES asks
// for the reconstruction. Returns nonzero on parameters the resolver cannot have
// produced.
int quant_sqrt_encode(void *restrict dst, const void *restrict src,
                      const quant_sqrt_params *p, int dtype, size_t nbElts);

// One pass over the raster, for the refusals and the floor flag. A sample that is
// not finite is counted and skipped; it has no place on the grid and the nodata
// codec in front of a lossy graph is what puts it back.
// Returns 0 when the raster holds a finite sample, 1 otherwise.
int quant_sqrt_scan(const void *src, int dtype, size_t nbElts,
                    quant_sqrt_stats *out);

#endif // GEOZL_CODECS_QUANT_SQRT_ENCODE_KERNEL_H
