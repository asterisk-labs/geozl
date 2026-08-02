#ifndef GEOZL_CODECS_QUANT_LINEAR_SPEC_H
#define GEOZL_CODECS_QUANT_LINEAR_SPEC_H

#include "geozl/quant_linear_params.h"

#include <stddef.h>

//   "LINEAR:MAX_ERROR=V"                 |x - x^| <= V
//   "LINEAR:MAX_ERROR=V,STORE=VALUES"    the stream carries the reconstruction
//
// Keys in any order, each once, unknown keys refused. See spec.md for what
// STORE=VALUES costs and when it is refused.
int quant_linear_parse(const char *s, quant_linear_spec *out, char *err,
                       size_t errSize);

// maxAbs and anyNegative come from quant_linear_scan.
int quant_linear_resolve(const quant_linear_spec *sp, int dtype, double maxAbs,
                         int anyNegative, quant_linear_params *out, char *err,
                         size_t errSize);

// The encoder's check on the frame it is about to write. worst comes back over
// the declared bound, so at or under one is the bound holding.
int quant_linear_verify(const void *src, const void *dec,
                        const quant_linear_spec *sp, int dtype, size_t nbElts,
                        double *worst);

#endif // GEOZL_CODECS_QUANT_LINEAR_SPEC_H
