#ifndef GEOZL_QUANT_LOG_PARAMS_H
#define GEOZL_QUANT_LOG_PARAMS_H

// No negative sample in the tile, so the decoder floors at zero.
#define QUANT_LOG_FLAG_NONNEGATIVE 1u

// The stream carries the reconstruction rather than the index.
#define QUANT_LOG_FLAG_STORE_VALUES 2u

#define QUANT_LOG_FLAGS_KNOWN                                                  \
  (QUANT_LOG_FLAG_NONNEGATIVE | QUANT_LOG_FLAG_STORE_VALUES)

// step is the width of one level in log2, so neighbouring levels differ by a
// factor of 2^step. The anchor is not here; it follows from the flags and the
// element type.
typedef struct {
  unsigned char flags;
  double step;
} quant_log_params;

typedef enum {
  QUANT_LOG_STORE_INDEX = 0,
  QUANT_LOG_STORE_VALUES = 1
} quant_log_store;

// The bound is kept apart from the resolved parameters because the encoder
// checks its own round trip against it after the grid is cut.
typedef struct {
  double rel_err;
  unsigned char store;
} quant_log_spec;

// What a pass over the tile reports. It decides refusals and the floor flag.
// Nothing here reaches a level, so two tiles that both resolve get the same grid.
typedef struct {
  double minAbs;    // smallest non-zero magnitude, or +inf if there is none
  double maxAbs;    // largest finite magnitude, or 0
  int anyNegative;
  int anySubnormal; // float types only
} quant_log_stats;

#endif // GEOZL_QUANT_LOG_PARAMS_H
