#ifndef GEOZL_CODECS_DECODE_QUANT_LOG_KERNEL_H
#define GEOZL_CODECS_DECODE_QUANT_LOG_KERNEL_H

#include "geozl/quant_log_params.h"

#include <stddef.h>

// Stream back into samples. dst holds nbElts elements at the width of dtype.
// Non-zero on parameters no frame this encoder wrote would carry.
int quant_log_decode(void *dst, const void *src, const quant_log_params *p,
                     int dtype, size_t nbElts);

#endif // GEOZL_CODECS_DECODE_QUANT_LOG_KERNEL_H
