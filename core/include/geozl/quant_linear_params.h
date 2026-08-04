#ifndef GEOZL_QUANT_LINEAR_PARAMS_H
#define GEOZL_QUANT_LINEAR_PARAMS_H

// The encoder found no negative sample, so the decoder floors at zero rather than
// at the type minimum. Measured, so a tile that does hold negatives keeps them.
#define QUANT_LINEAR_FLAG_NONNEGATIVE 1u

// The stream carries the reconstruction, not the index, so the decoder does not
// multiply.
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

// The bound travels alongside the resolved parameters because the encoder measures
// its own round trip against it, and the parameters stop carrying it once the grid
// is cut.
typedef struct {
  double max_error;
  unsigned char store;
} quant_linear_spec;

// What a pass over the tile reports. It decides the refusal and the floor flag.
// Nothing here reaches a level, so two tiles that both resolve get the same
// grid.
typedef struct {
  double maxAbs; // largest finite magnitude, or 0
  int anyNegative;
} quant_linear_stats;

#endif // GEOZL_QUANT_LINEAR_PARAMS_H
