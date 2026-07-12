#ifndef GEOZL_CODECS_DELTA_N_DECODE_KERNEL_H
#define GEOZL_CODECS_DELTA_N_DECODE_KERNEL_H

#include <stddef.h> // size_t

// Inverse vertical predictor. Row major plane, @width samples per row,
// @nbElts total, @eltWidth bytes per sample. The first row is absolute, every
// later row is the row above plus the residual. No scan, a plain vector add
// across the row, all columns independent. @dst and @src must not alias, the
// row above is read from @dst while the residual is read from @src.
void delta_n_decode(void *dst, const void *src, size_t width, size_t nbElts,
                    size_t eltWidth);

#endif // GEOZL_CODECS_DELTA_N_DECODE_KERNEL_H
