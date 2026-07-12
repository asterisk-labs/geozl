#include "encode_intmult_kernel.h"

#include <stdint.h>

// div + mod fuse to a single idiv on x86_64 and most targets; base is a
// runtime divisor
#define INTMULT_SPLIT(T)                                                       \
  do {                                                                         \
    const T *s = (const T *)src;                                               \
    T *m = (T *)mults;                                                         \
    T *a = (T *)adjs;                                                          \
    const T b = (T)base;                                                       \
    for (size_t k = 0; k < nb_elts; ++k) {                                     \
      const T u = s[k];                                                        \
      m[k] = (T)(u / b);                                                       \
      a[k] = (T)(u % b);                                                       \
    }                                                                          \
  } while (0)

void intmult_split(void *mults, void *adjs, const void *src, size_t nb_elts,
                   size_t elt_width, uint64_t base) {
  switch (elt_width) {
  case 1:
    INTMULT_SPLIT(uint8_t);
    break;
  case 2:
    INTMULT_SPLIT(uint16_t);
    break;
  case 4:
    INTMULT_SPLIT(uint32_t);
    break;
  case 8:
    INTMULT_SPLIT(uint64_t);
    break;
  default:
    break;
  }
}
