#ifndef GEOZL_CODECS_QUANT_SQRT_FIT_H
#define GEOZL_CODECS_QUANT_SQRT_FIT_H

#include "geozl/quant_sqrt_params.h"

#include <stddef.h>

// sigma^2 = a + b*x, fitted blind from the raster itself, for a recipe that
// carries no A and B.
//
// The method is the scatterplot of local mean against local variance with a low
// quantile per intensity bin, after Abramova and others, SPIE 10004, 2016. The
// same model is Foi, Trimeche, Katkovnik and Egiazarian, IEEE TIP 17(10), 2008.
//
// The fit reads the raster, so it moves the grid. That is fine as long as it runs
// once per product rather than once per tile, and this file does not enforce that
// because it cannot see how the caller cuts the data. What it does instead is
// report how much the fit should be believed, in bins, range, colin and resid, so
// the caller can pool several rasters through the accumulator and decide.

// One raster in one call. width * height has to equal the element count.
int quant_sqrt_fit(const void *src, int dtype, size_t width, size_t height,
                   quant_sqrt_noise *out, char *err, size_t errSize);

// The same thing over several rasters. The block statistics are pooled before
// anything is fitted, which is not the same as averaging separate fits: a tile
// that covers one end of the intensity range and a tile that covers the other
// give a usable curve together and neither gives one alone.
typedef struct {
  double *mu;      // local means
  double *s2;      // local variances, already corrected for the quantile bias
  size_t n;        // pairs held
  size_t cap;      // pairs the buffer holds
  size_t stride;   // one block in every stride is kept once the cap is reached
  size_t seen;     // blocks measured, including the ones dropped
  int failed;      // allocation gave out
} quant_sqrt_accum;

void quant_sqrt_accum_init(quant_sqrt_accum *acc);
void quant_sqrt_accum_free(quant_sqrt_accum *acc);
int quant_sqrt_accum_push(quant_sqrt_accum *acc, const void *src, int dtype,
                          size_t width, size_t height);
int quant_sqrt_accum_solve(const quant_sqrt_accum *acc, quant_sqrt_noise *out,
                           char *err, size_t errSize);

#endif // GEOZL_CODECS_QUANT_SQRT_FIT_H
