#ifndef GEOZL_CODECS_QUANT_LOG_SPEC_H
#define GEOZL_CODECS_QUANT_LOG_SPEC_H

#include "geozl/quant_log_params.h"

#include <stddef.h>

// "LOG:MAX_ERROR=1%" or "LOG:MAX_ERROR=1%,STORE=VALUES".
int quant_log_parse(const char *s, quant_log_spec *out, char *err,
                    size_t errSize);

// Spec into the parameters a frame carries. sc decides refusals only; no number
// in it reaches a level.
int quant_log_resolve(const quant_log_spec *sp, int dtype,
                      const quant_log_stats *sc, quant_log_params *out,
                      char *err, size_t errSize);

// The encoder's check on the frame it is about to write. worst comes back as the
// error over the bound that was declared, so at or under one is the bound
// holding. skipped counts samples below the smallest normal of a float type,
// where a relative bound is not something the type can carry.
int quant_log_verify(const void *src, const void *dec, const quant_log_spec *sp,
                     int dtype, size_t nbElts, double *worst, size_t *skipped);

#endif // GEOZL_CODECS_QUANT_LOG_SPEC_H
