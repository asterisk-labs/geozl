#include "encode_nodata_kernel.h"

#include <stdint.h>
#include <string.h>

// Exponent all ones with a nonzero mantissa is a NaN at every IEEE width. An
// infinity has the same exponent and a zero mantissa, and stays a value.
#define NODATA_ISNAN16(b) (((b) & 0x7C00u) == 0x7C00u && ((b) & 0x03FFu) != 0)
#define NODATA_ISNAN32(b) \
  (((b) & 0x7F800000u) == 0x7F800000u && ((b) & 0x007FFFFFu) != 0)
#define NODATA_ISNAN64(b)                              \
  (((b) & 0x7FF0000000000000ull) == 0x7FF0000000000000ull \
   && ((b) & 0x000FFFFFFFFFFFFFull) != 0)

#define NODATA_FIND_NAN(T, TEST)                                               \
  do {                                                                         \
    const T *s = (const T *)src;                                               \
    for (size_t i = 0; i < nb_elts; ++i) {                                     \
      if (TEST(s[i])) {                                                        \
        *pattern = (uint64_t)s[i];                                             \
        return 1;                                                              \
      }                                                                        \
    }                                                                          \
    return 0;                                                                  \
  } while (0)

int nodata_find_nan(uint64_t *pattern, const void *src, size_t nb_elts,
                    size_t elt_width) {
  switch (elt_width) {
  case 2:
    NODATA_FIND_NAN(uint16_t, NODATA_ISNAN16);
  case 4:
    NODATA_FIND_NAN(uint32_t, NODATA_ISNAN32);
  case 8:
    NODATA_FIND_NAN(uint64_t, NODATA_ISNAN64);
  default:
    return 0; // no IEEE type is one byte wide
  }
}

#define NODATA_MARK_NAN(T, TEST)                                               \
  do {                                                                         \
    const T *s = (const T *)src;                                               \
    size_t n = 0;                                                              \
    for (size_t i = 0; i < nb_elts; ++i) {                                     \
      const int hit = TEST(s[i]);                                              \
      mask[i] = hit ? GEOZL_NODATA_INVALID : GEOZL_NODATA_VALID;               \
      n += (size_t)hit;                                                        \
    }                                                                          \
    return n;                                                                  \
  } while (0)

size_t nodata_mark_nan(uint8_t *mask, const void *src, size_t nb_elts,
                       size_t elt_width) {
  switch (elt_width) {
  case 2:
    NODATA_MARK_NAN(uint16_t, NODATA_ISNAN16);
  case 4:
    NODATA_MARK_NAN(uint32_t, NODATA_ISNAN32);
  case 8:
    NODATA_MARK_NAN(uint64_t, NODATA_ISNAN64);
  default:
    memset(mask, GEOZL_NODATA_VALID, nb_elts);
    return 0;
  }
}

#define NODATA_MARK_VALUE(T)                                                   \
  do {                                                                         \
    const T *s = (const T *)src;                                               \
    const T p = (T)pattern;                                                    \
    size_t n = 0;                                                              \
    for (size_t i = 0; i < nb_elts; ++i) {                                     \
      const int hit = (s[i] == p);                                             \
      mask[i] = hit ? GEOZL_NODATA_INVALID : GEOZL_NODATA_VALID;               \
      n += (size_t)hit;                                                        \
    }                                                                          \
    return n;                                                                  \
  } while (0)

size_t nodata_mark_value(uint8_t *mask, const void *src, size_t nb_elts,
                         size_t elt_width, uint64_t pattern) {
  switch (elt_width) {
  case 1:
    NODATA_MARK_VALUE(uint8_t);
  case 2:
    NODATA_MARK_VALUE(uint16_t);
  case 4:
    NODATA_MARK_VALUE(uint32_t);
  case 8:
    NODATA_MARK_VALUE(uint64_t);
  default:
    return 0;
  }
}

// A hole takes the last valid sample of its row. A hole that opens a row takes
// the sample above, already filled because rows run top to bottom, so a whole
// leading row of holes still comes out flat rather than stepping off zero.
#define NODATA_FILL(T)                                                         \
  do {                                                                         \
    T *d = (T *)dst;                                                           \
    const T *s = (const T *)src;                                               \
    for (size_t off = 0; off < nb_elts; off += w) {                            \
      T last = 0;                                                              \
      int seen = 0;                                                            \
      for (size_t c = 0; c < w; ++c) {                                         \
        const size_t i = off + c;                                              \
        if (mask[i] != GEOZL_NODATA_INVALID) {                                 \
          last = s[i];                                                         \
          seen = 1;                                                            \
          d[i] = last;                                                         \
        } else if (seen) {                                                     \
          d[i] = last;                                                         \
        } else if (off > 0) {                                                  \
          d[i] = d[i - w];                                                     \
        } else {                                                               \
          d[i] = 0;                                                            \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  } while (0)

void nodata_fill(void *dst, const void *src, const uint8_t *mask, size_t width,
                 size_t nb_elts, size_t elt_width) {
  if (nb_elts == 0)
    return;
  // An unusable width degrades to one row, which is still a correct fill.
  const size_t w =
      (width == 0 || width > nb_elts || nb_elts % width != 0) ? nb_elts : width;
  switch (elt_width) {
  case 1:
    NODATA_FILL(uint8_t);
    break;
  case 2:
    NODATA_FILL(uint16_t);
    break;
  case 4:
    NODATA_FILL(uint32_t);
    break;
  case 8:
    NODATA_FILL(uint64_t);
    break;
  default:
    break; // rejected by the binding
  }
}
