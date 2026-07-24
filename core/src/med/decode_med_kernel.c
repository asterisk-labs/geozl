// Inverse MED predictor, JPEG-LS (ITU-T T.87 A.4.2,
// https://www.itu.int/rec/T-REC-T.87/en). Row zero and
// column zero are peeled so the interior loop carries no boundary test.

#include "decode_med_kernel.h"

#include "common/raster.h" // geozl_row_width

#include <stdint.h>

#define MED_DEC(T)                                                             \
  do {                                                                         \
    T *d = (T *)dst;                                                           \
    const T *s = (const T *)src;                                               \
    const size_t rows = nbElts / w;                                            \
    d[0] = s[0];                                                               \
    for (size_t c = 1; c < w; ++c)                                             \
      d[c] = (T)(s[c] + d[c - 1]);                                             \
    for (size_t r = 1; r < rows; ++r) {                                        \
      const size_t row = r * w;                                                \
      d[row] = (T)(s[row] + d[row - w]);                                       \
      for (size_t c = 1; c < w; ++c) {                                         \
        const size_t i = row + c;                                              \
        const T Wv = d[i - 1];                                                 \
        const T Nv = d[i - w];                                                 \
        const T NWv = d[i - w - 1];                                            \
        const T mn = Wv < Nv ? Wv : Nv;                                        \
        const T mx = Wv < Nv ? Nv : Wv;                                        \
        const T P = (NWv >= mx) ? mn : (NWv <= mn) ? mx : (T)(Wv + Nv - NWv);  \
        d[i] = (T)(s[i] + P);                                                  \
      }                                                                        \
    }                                                                          \
  } while (0)

int med_decode(void *dst, const void *src, size_t width, size_t nbElts,
               size_t eltWidth) {
  // A width that does not divide nbElts would leave the last row short,
  // and the row loops below assume every row is complete.
  const size_t w = geozl_row_width(width, nbElts);
  if (w == 0)
    return 1;
  switch (eltWidth) {
  case 1:
    MED_DEC(uint8_t);
    break;
  case 2:
    MED_DEC(uint16_t);
    break;
  case 4:
    MED_DEC(uint32_t);
    break;
  case 8:
    MED_DEC(uint64_t);
    break;
  default:
    return 1; // eltWidth must be 1, 2, 4 or 8
  }
  return 0;
}

#undef MED_DEC