#ifndef GEOZL_QUANT_PARAMS_H
#define GEOZL_QUANT_PARAMS_H

#include <stdint.h>

// A quantizer that respects a pointwise bound b(x) is a uniform quantizer in a
// warped domain w, with w' = 1/b. Each curve is that integral for one way the
// measurement error of the data grows with the value:
//
//   linear   b constant        w = x           instrument error, elevation
//   sqrt     b ~ sqrt(x)       w = 2*sqrt(x)   photon counting, optical
//   log      b ~ x             w = ln(x)       multiplicative, SAR speckle
//
// so the curve is not a free choice, it follows from the sensor.
typedef enum {
  QUANT_CURVE_LINEAR = 0,
  QUANT_CURVE_SQRT = 1,
  QUANT_CURVE_LOG = 2
} quant_curve;

// The stream holds reconstructed values rather than indices, so the decoder
// only copies. The linear curve on an integer type only, elsewhere the
// reconstruction is not the integer stream the codec emits.
#define QUANT_FLAG_STORE_VALUES 1u

// step is the quantization step in the warped domain, and zero is an exact
// passthrough. offset anchors the curve: the noise floor over the gain for
// sqrt, the smallest magnitude the geometric grid serves for log. nsub is the
// number of leading indices the log curve carries exactly, for the range where
// the representable values sit too far apart for any grid to meet the bound.
typedef struct {
  uint8_t curve;
  uint8_t flags;
  double step;
  double offset;
  uint64_t nsub;
} quant_params;

// The error argument of the 2d API, parsed. The bound it declares is what the
// encoder measures its own round trip against, so it travels alongside the
// resolved parameters, which no longer carry it once the grid has been cut.
typedef enum {
  QUANT_SPEC_LOSSLESS = 0,
  QUANT_SPEC_EXPLICIT = 1
} quant_spec_mode;

typedef struct {
  unsigned char mode;
  unsigned char curve;
  double abs_err;
  double rel_err;
  double shot_a, shot_b, shot_k;
} quant_spec;

#endif // GEOZL_QUANT_PARAMS_H