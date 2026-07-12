#ifndef GEOZL_CODECS_FLOATQUANT_DECODE_KERNEL_H
#define GEOZL_CODECS_FLOATQUANT_DECODE_KERNEL_H

#include <stddef.h>
#include <stdint.h>

// Rebuild the float stream from the high part and the low k mantissa bits.
// dst holds nb_elts elements of elt_width bytes. elt_width must be 4 or 8. k
// must be in 1..=PRECISION_BITS. Returns 0 on success, nonzero if any secondary
// is not a k-bit value, the only corruption the pairing can detect. On nonzero
// dst holds garbage but every access stayed in bounds.
int floatquant_join(void *dst, const void *primary, const void *secondary,
                    size_t nb_elts, size_t elt_width, unsigned k);

#endif // GEOZL_CODECS_FLOATQUANT_DECODE_KERNEL_H
