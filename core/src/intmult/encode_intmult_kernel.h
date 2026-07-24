// Derived from pcodec, https://github.com/pcodec/pcodec, Apache License 2.0.
// Full licence text in LICENSE.pcodec at the root of this repository.
// Ported to C from pcodec's IntMult mode. The port is partial and geozl does
// not reproduce pcodec's wire format.

#ifndef GEOZL_CODECS_INTMULT_ENCODE_KERNEL_H
#define GEOZL_CODECS_INTMULT_ENCODE_KERNEL_H

#include <stddef.h>
#include <stdint.h>

// Split into (mult, adj) against an integer base, per element u of elt_width bytes:
//   mult = u / base
//   adj  = u % base
// mults and adjs each hold nb_elts elements of elt_width bytes. base >= 2.
// elt_width must be 1, 2, 4 or 8.
void intmult_split(void *mults, void *adjs, const void *src, size_t nb_elts,
                   size_t elt_width, uint64_t base);

#endif // GEOZL_CODECS_INTMULT_ENCODE_KERNEL_H
