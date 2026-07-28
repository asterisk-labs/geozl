#ifndef GEOZL_CODECS_QUANT_DTYPE_H
#define GEOZL_CODECS_QUANT_DTYPE_H

// Element type of the original stream, carried in the codec header so the
// decoder rebuilds it. The index stream between encode and decode is always an
// integer, the original type only matters at the two ends. The value is the
// wire code, frozen.
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

#endif // GEOZL_CODECS_QUANT_DTYPE_H
