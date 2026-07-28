#ifndef GEOZL_CODECS_ENCODE_QUANT_KERNEL_H
#define GEOZL_CODECS_ENCODE_QUANT_KERNEL_H

#include "quant_curve.h" // quant_params
#include "quant_dtype.h" // quant_dtype

#include <stddef.h>

// Quantize nbElts samples of dtype into the index stream at dst. The index
// keeps the sample width, signed for a float original. Returns 0, or nonzero
// when dtype or the curve is out of range, or when a stored reconstruction is
// asked for on a float type.
int quant_encode(void *dst, const void *src, const quant_params *p, int dtype,
                 size_t nbElts);

// Smallest and largest non-zero magnitude, and whether any sample is negative.
// The log curve anchors its grid on the smallest magnitude in the tile, so both
// the explicit and the trained paths need this before they can finish the
// parameters. NaN and infinity are skipped, the nodata codec handles those.
// Returns 0 when the tile holds at least one finite non-zero sample.
int quant_scan(const void *src, int dtype, size_t nbElts, double *minAbs,
               double *maxAbs, int *anyNegative);

#endif // GEOZL_CODECS_ENCODE_QUANT_KERNEL_H
