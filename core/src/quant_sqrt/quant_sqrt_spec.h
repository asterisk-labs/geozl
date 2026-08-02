#ifndef GEOZL_CODECS_QUANT_SQRT_SPEC_H
#define GEOZL_CODECS_QUANT_SQRT_SPEC_H

#include "geozl/quant_sqrt_params.h"

#include <stddef.h>

//   "SQRT:MAX_ERROR=0.5N"                   |x - x^| <= 0.5 * sqrt(a + b*x)
//   "SQRT:MAX_ERROR=0.5N,A=100,B=1"         the same, with the curve given
//   "SQRT:MAX_ERROR=0.5N,STORE=VALUES"      the stream carries the reconstruction
//
// MAX_ERROR is in units of the noise sigma and the N is required. Without it a
// MAX_ERROR of 0.5 reads as half a count, which is a plausible thing to mean and
// would quantize far finer than the caller asked for, silently.
//
// A and B are optional and travel together. Neither means the caller has to hand
// quant_sqrt_resolve a fit; one without the other is refused rather than half
// taken.
int quant_sqrt_parse(const char *s, quant_sqrt_spec *out, char *err,
                     size_t errSize);

// The curve into the grid. ft may be NULL when the recipe carries A and B, and
// has to be a fit that succeeded when it does not.
int quant_sqrt_resolve(const quant_sqrt_spec *sp, int dtype,
                       const quant_sqrt_stats *sc, const quant_sqrt_noise *ft,
                       quant_sqrt_params *out, char *err, size_t errSize);

// The bound the resolved parameters hold at x, so a caller can report it without
// knowing how the grid was cut.
double quant_sqrt_bound(const quant_sqrt_spec *sp, const quant_sqrt_noise *ft,
                        double x);

// The encoder's check on the frame it is about to write. worst comes back as the
// error over the bound that was declared, so at or under one is the bound
// holding.
int quant_sqrt_verify(const void *src, const void *dec,
                      const quant_sqrt_spec *sp, const quant_sqrt_noise *ft,
                      int dtype, size_t nbElts, double *worst);

#endif // GEOZL_CODECS_QUANT_SQRT_SPEC_H
