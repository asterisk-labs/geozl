#ifndef GEOZL_DTYPE_H
#define GEOZL_DTYPE_H

#include <stddef.h>

// The element type of the original raster. It crosses the C API as an int and
// every lossy codec writes it into its header, so these numbers are frozen.
//
// Each codec folder keeps its own copy of this enum, on the same numbers, so it
// lifts out of the tree whole. This is the copy a caller sees.
typedef enum {
  GEOZL_DT_U8 = 0,
  GEOZL_DT_U16 = 1,
  GEOZL_DT_U32 = 2,
  GEOZL_DT_U64 = 3,
  GEOZL_DT_I8 = 4,
  GEOZL_DT_I16 = 5,
  GEOZL_DT_I32 = 6,
  GEOZL_DT_I64 = 7,
  GEOZL_DT_F16 = 8,
  GEOZL_DT_F32 = 9,
  GEOZL_DT_F64 = 10
} geozl_dtype;

#define GEOZL_DT_LAST_INT GEOZL_DT_I64
#define GEOZL_DT_OK(d) ((d) >= GEOZL_DT_U8 && (d) <= GEOZL_DT_F64)

// The codes run without gaps, so a code indexes this directly. GEOZL_DT_OK is
// the caller's job, and everything that calls this has already done it.
static inline size_t geozl_dtype_width(int dtype) {
  static const size_t w[] = {1, 2, 4, 8, 1, 2, 4, 8, 2, 4, 8};
  return w[dtype];
}

#endif // GEOZL_DTYPE_H