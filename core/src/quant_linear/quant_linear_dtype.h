#ifndef GEOZL_CODECS_QUANT_LINEAR_DTYPE_H
#define GEOZL_CODECS_QUANT_LINEAR_DTYPE_H

// Element type of the original stream, carried in the codec header so the decoder
// rebuilds it. The index stream between encode and decode is always a signed
// integer, the original type only matters at the two ends. The value is the wire
// code, frozen.
typedef enum {
    QL_U8  = 0,
    QL_U16 = 1,
    QL_U32 = 2,
    QL_U64 = 3,
    QL_I8  = 4,
    QL_I16 = 5,
    QL_I32 = 6,
    QL_I64 = 7,
    QL_F16 = 8,
    QL_F32 = 9,
    QL_F64 = 10
} ql_dtype;

#endif // GEOZL_CODECS_QUANT_LINEAR_DTYPE_H
