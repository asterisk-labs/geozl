#include "encode_planar_zigzag_pfor_kernel.h"

#include "common/raster.h"
#include "pfor/encode_pfor_kernel.h"
#include "pfor/pfor_check.h"

#include <stdint.h>
#include <string.h>

size_t planar_zigzag_pfor_bound(size_t nbElts, size_t eltWidth) {
  return pfor_bound(nbElts, eltWidth);
}

#define PLANAR_ZIGZAG_PFOR_ENC(T, B)                                          \
  do {                                                                         \
    const T *s = (const T *)src;                                               \
    T *z = (T *)scratch;                                                       \
    for (size_t off = 0; off < nbElts; off += GEOZL_PFOR_BLOCK) {              \
      const size_t have = nbElts - off;                                        \
      const size_t n =                                                         \
          (have < GEOZL_PFOR_BLOCK) ? have : GEOZL_PFOR_BLOCK;                 \
      for (size_t i = 0; i < n; ++i) {                                        \
        const size_t pos = off + i;                                            \
        const size_t inPlane = pos % planeElts;                                \
        const size_t column = inPlane % width;                                 \
        const size_t row = inPlane / width;                                    \
        const T Wv = (column > 0) ? s[pos - 1] : 0;                            \
        const T Nv = (row > 0) ? s[pos - width] : 0;                           \
        const T NWv =                                                          \
            (row > 0 && column > 0) ? s[pos - width - 1] : 0;                  \
        const T residual = (T)(s[pos] - (T)(Wv + Nv - NWv));                   \
        const T sign = (T)(0 - (residual >> ((B)-1)));                         \
        z[i] = (T)((T)(residual << 1) ^ sign);                                 \
      }                                                                        \
      if (n != GEOZL_PFOR_BLOCK)                                               \
        memset(z + n, 0, (GEOZL_PFOR_BLOCK - n) * sizeof(T));                  \
      const size_t used = pfor_encode_block(p, z, eltWidth);                   \
      if (used == 0)                                                           \
        return 1;                                                              \
      p += used;                                                               \
    }                                                                          \
  } while (0)

int planar_zigzag_pfor_encode(void *dst, size_t dstCapacity, size_t *outSize,
                              const void *src, size_t width, size_t nbElts,
                              size_t eltWidth, uint32_t planes) {
  if (dst == NULL || src == NULL || outSize == NULL || planes == 0 ||
      nbElts == 0 || nbElts % planes != 0)
    return 1;
  const size_t planeElts = nbElts / planes;
  if (geozl_row_width(width, planeElts) == 0)
    return 1;
  const size_t bound = planar_zigzag_pfor_bound(nbElts, eltWidth);
  if (bound == 0 || dstCapacity < bound)
    return 1;

  uint8_t *p = (uint8_t *)dst;
  uint64_t scratch[GEOZL_PFOR_BLOCK];
  switch (eltWidth) {
  case 1:
    PLANAR_ZIGZAG_PFOR_ENC(uint8_t, 8);
    break;
  case 2:
    PLANAR_ZIGZAG_PFOR_ENC(uint16_t, 16);
    break;
  case 4:
    PLANAR_ZIGZAG_PFOR_ENC(uint32_t, 32);
    break;
  case 8:
    PLANAR_ZIGZAG_PFOR_ENC(uint64_t, 64);
    break;
  default:
    return 1;
  }
  *outSize = (size_t)(p - (uint8_t *)dst);
  return 0;
}

#undef PLANAR_ZIGZAG_PFOR_ENC
