#include "decode_binoffset_kernel.h"

#include <stdint.h>

// lower = (bin != 0) << ((bin - 1) & (TBITS - 1))   mask keeps a corrupt bin's
//                                                    shift in range
// valid: off < max(lower, 1)
#define BINOFFSET_JOIN(T, TBITS)                                               \
  do {                                                                         \
    const T *o = (const T *)offsets;                                           \
    T *d = (T *)dst;                                                           \
    unsigned bad = 0;                                                          \
    for (size_t k = 0; k < nb_elts; ++k) {                                     \
      const unsigned bl = bins[k];                                             \
      const T off = o[k];                                                      \
      const T lower = (T)((T)(bl != 0u) << ((bl - 1u) & (TBITS - 1u)));        \
      bad |= (unsigned)(bl > TBITS);                                           \
      bad |= (unsigned)(off >= (T)(lower | (T)(lower == 0u)));                 \
      d[k] = (T)(lower | off);                                                 \
    }                                                                          \
    return (int)bad;                                                           \
  } while (0)

int binoffset_join(void *dst, const uint8_t *bins, const void *offsets,
                   size_t nb_elts, size_t elt_width) {
  switch (elt_width) {
  case 1:
    BINOFFSET_JOIN(uint8_t, 8u);
  case 2:
    BINOFFSET_JOIN(uint16_t, 16u);
  case 4:
    BINOFFSET_JOIN(uint32_t, 32u);
  case 8:
    BINOFFSET_JOIN(uint64_t, 64u);
  default:
    return 1;
  }
}

// mask = low offset_bits[b] bits; padded entries have offset_bits 0, so any
// nonzero offset there is rejected and their bin ids already set the flag
#define BINOFFSET_JOIN_TABLE(T, TBITS)                                         \
  do {                                                                         \
    const T *o = (const T *)offsets;                                           \
    T *d = (T *)dst;                                                           \
    unsigned bad = 0;                                                          \
    for (size_t k = 0; k < nb_elts; ++k) {                                     \
      const unsigned b = bins[k];                                              \
      const T off = o[k];                                                      \
      const unsigned ob = offset_bits[b];                                      \
      const T mask = (T)(                                                      \
          (ob >= TBITS) ? (T)-1 : (T)(((T)1 << (ob & (TBITS - 1u))) - 1u));    \
      bad |= (unsigned)(b >= nb_bins);                                         \
      bad |= (unsigned)((T)(off & (T)~mask) != 0u);                            \
      d[k] = (T)((T)lowers[b] + off);                                          \
    }                                                                          \
    return (int)bad;                                                           \
  } while (0)

int binoffset_join_table(void *dst, const uint8_t *bins, const void *offsets,
                         size_t nb_elts, size_t elt_width,
                         const uint64_t lowers[256],
                         const uint8_t offset_bits[256], unsigned nb_bins) {
  switch (elt_width) {
  case 1:
    BINOFFSET_JOIN_TABLE(uint8_t, 8u);
  case 2:
    BINOFFSET_JOIN_TABLE(uint16_t, 16u);
  case 4:
    BINOFFSET_JOIN_TABLE(uint32_t, 32u);
  case 8:
    BINOFFSET_JOIN_TABLE(uint64_t, 64u);
  default:
    return 1;
  }
}
