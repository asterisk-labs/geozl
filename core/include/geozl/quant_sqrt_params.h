#ifndef GEOZL_QUANT_SQRT_PARAMS_H
#define GEOZL_QUANT_SQRT_PARAMS_H

// No sample below zero, so the decoder floors there. Index zero rebuilds to
// -offset, which is below zero wherever the curve has a noise floor, and on an
// unsigned type that would wrap.
#define QUANT_SQRT_FLAG_NONNEGATIVE 1u

// The stream carries the reconstruction rather than the index.
#define QUANT_SQRT_FLAG_STORE_VALUES 2u

// The decoder rebuilds in float rather than double. Both are plain IEEE, so both
// give the same bits on every machine; the bit says which one the step was cut
// against, so a reader that took the other would hold a bound the frame does not
// declare. Float32 output only.
#define QUANT_SQRT_FLAG_DECODE_F32 4u

#define QUANT_SQRT_FLAGS_KNOWN                                                 \
  (QUANT_SQRT_FLAG_NONNEGATIVE | QUANT_SQRT_FLAG_STORE_VALUES |                \
   QUANT_SQRT_FLAG_DECODE_F32)

// The grid is uniform in u = sqrt(x + offset), so step is the width of one level
// in u and offset is where the curve is anchored. Both come from the recipe and
// the type; neither is measured off the raster.
typedef struct {
  unsigned char flags;
  double step;
  double offset;
} quant_sqrt_params;

typedef enum {
  QUANT_SQRT_STORE_INDEX = 0,
  QUANT_SQRT_STORE_VALUES = 1
} quant_sqrt_store;

// "SQRT:MAX_ERROR=0.5N" holds |x - x^| <= 0.5 * sqrt(a + b*x). With no A and B in
// the recipe the two come from quant_sqrt_fit and have_ab stays clear, which is
// what the resolver checks before it asks for one.
typedef struct {
  double k;
  double a;
  double b;
  unsigned char store;
  unsigned char have_ab;
} quant_sqrt_spec;

// One pass over the raster. Decides refusals and the floor flag. No number here
// reaches a level, so two rasters that both resolve share a grid.
typedef struct {
  double lo;        // smallest sample, signed, +inf when there is none finite
  double hi;        // largest sample, signed, -inf when there is none finite
  int anyNegative;
  int anyNonFinite;
} quant_sqrt_stats;

// What the blind fit reports. a and b are the model; the rest is how much the
// caller should believe it. Pooling several rasters is the caller's job, and
// quant_sqrt_accum is there for it.
typedef struct {
  double a;
  double b;
  int ok;
  int blocks;      // blocks measured
  int bins;        // intensity bins that survived
  double range;    // mu_max / mu_min over the surviving bins
  double colin;    // mean mu over its spread; large means a and b are collinear
  double resid;    // relative rms of the fit residuals
} quant_sqrt_noise;

#endif // GEOZL_QUANT_SQRT_PARAMS_H
