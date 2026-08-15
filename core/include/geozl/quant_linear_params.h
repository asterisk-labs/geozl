#ifndef GEOZL_QUANT_LINEAR_PARAMS_H
#define GEOZL_QUANT_LINEAR_PARAMS_H

// Clamp reconstruction at zero when the input has no negative samples.
#define QUANT_LINEAR_FLAG_NONNEGATIVE 1u

// Store reconstructed values instead of quantizer indices.
#define QUANT_LINEAR_FLAG_STORE_VALUES 2u

#define QUANT_LINEAR_FLAGS_KNOWN                                               \
  (QUANT_LINEAR_FLAG_NONNEGATIVE | QUANT_LINEAR_FLAG_STORE_VALUES)

typedef struct {
  unsigned char flags;
  double step;
} quant_linear_params;

typedef enum {
  QUANT_LINEAR_STORE_INDEX = 0,
  QUANT_LINEAR_STORE_VALUES = 1
} quant_linear_store;

// Parsed recipe before it is resolved for a dtype and raster.
typedef struct {
  double max_error;
  unsigned char store;
} quant_linear_spec;

// Statistics used to resolve a recipe.
typedef struct {
  double maxAbs; // largest finite magnitude, or 0
  int anyNegative;
} quant_linear_stats;

#endif // GEOZL_QUANT_LINEAR_PARAMS_H
