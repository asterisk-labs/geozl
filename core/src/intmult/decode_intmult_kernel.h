#ifndef GEOZL_CODECS_INTMULT_DECODE_KERNEL_H
#define GEOZL_CODECS_INTMULT_DECODE_KERNEL_H

#include <stddef.h>
#include <stdint.h>

// u = mult * base + adj (wrapping in elt_width), for nb_elts elements of
// elt_width bytes (1, 2, 4 or 8). base >= 2. Returns nonzero if any adj is not
// below base (the only detectable corruption). On nonzero, dst is garbage but
// every access stayed in bounds; the multiply wraps for a corrupt mult,
// deterministic and memory safe.
int intmult_join(void *dst, const void *mults, const void *adjs, size_t nb_elts,
                 size_t elt_width, uint64_t base);

#endif // GEOZL_CODECS_INTMULT_DECODE_KERNEL_H
