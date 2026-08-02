#ifndef GEOZL_CODECS_ENCODE_QUANT_LOG_KERNEL_H
#define GEOZL_CODECS_ENCODE_QUANT_LOG_KERNEL_H

#include "geozl/quant_log_params.h"

#include <stddef.h>

// Samples into the stream. dst is a separate buffer of nbElts elements at the
// width of dtype. Non-zero on a parameter block no frame this codec writes
// would carry.
int quant_log_encode(void *restrict dst, const void *restrict src,
                     const quant_log_params *p, int dtype, size_t nbElts);

// One pass over the tile, for the refusals and the floor flag.
int quant_log_scan(const void *src, int dtype, size_t nbElts,
                   quant_log_stats *out);

#endif // GEOZL_CODECS_ENCODE_QUANT_LOG_KERNEL_H
