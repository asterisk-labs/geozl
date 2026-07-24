#ifndef GEOZL_CODECS_QUANT_LINEAR_DECODE_KERNEL_H
#define GEOZL_CODECS_QUANT_LINEAR_DECODE_KERNEL_H

#include <stddef.h>

// Rebuild the original typed stream from integer indices, x = q * scale. Not a
// true inverse, the loss happened in encode. dtype is the ql_dtype of the
// output, scale = 2 * max_error. dst holds nbElts of the output element width.
// Returns 0 on success, nonzero when dtype is not a valid ql_dtype. On a
// nonzero return dst is left untouched.
int quant_linear_decode(void *dst, const void *src, double scale, int dtype,
                        size_t nbElts);

#endif // GEOZL_CODECS_QUANT_LINEAR_DECODE_KERNEL_H
