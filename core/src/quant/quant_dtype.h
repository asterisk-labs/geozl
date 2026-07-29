#ifndef GEOZL_CODECS_QUANT_DTYPE_H
#define GEOZL_CODECS_QUANT_DTYPE_H

#include <stddef.h>

// Original element type, carried in the codec header. The stream between the
// two ends is always integer, so this only matters there. Wire codes, frozen.
typedef enum {
  Q_U8 = 0,
  Q_U16 = 1,
  Q_U32 = 2,
  Q_U64 = 3,
  Q_I8 = 4,
  Q_I16 = 5,
  Q_I32 = 6,
  Q_I64 = 7,
  Q_F16 = 8,
  Q_F32 = 9,
  Q_F64 = 10
} quant_dtype;

#define Q_LAST_INT Q_I64

// Width of the type a code names. The codes run 0 to Q_F64 with no gaps, so a
// code indexes this directly, and every caller checks that range first.
static inline size_t quant_width(int dtype) {
  static const size_t w[] = {1, 2, 4, 8, 1, 2, 4, 8, 2, 4, 8};
  return w[dtype];
}

#endif // GEOZL_CODECS_QUANT_DTYPE_H
