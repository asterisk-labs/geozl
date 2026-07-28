#ifndef GEOZL_CODECS_DECODE_QUANT_KERNEL_H
#define GEOZL_CODECS_DECODE_QUANT_KERNEL_H

#include "quant_curve.h" // quant_params
#include "quant_dtype.h" // quant_dtype

#include <stddef.h>

// Rebuild nbElts samples of dtype from the index stream at src. Returns 0, or
// nonzero when dtype or the curve is out of range.
int quant_decode(void *dst, const void *src, const quant_params *p, int dtype,
                 size_t nbElts);

#endif // GEOZL_CODECS_DECODE_QUANT_KERNEL_H
