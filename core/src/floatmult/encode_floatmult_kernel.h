// Derived from pcodec, https://github.com/pcodec/pcodec, Apache License 2.0.
// Full licence text in LICENSE.pcodec at the root of this repository.
// Ported to C from pcodec's FloatMult mode. The latent mapping lives in
// floatmult_common.h. The port is partial and geozl does not reproduce
// pcodec's wire format.

#ifndef GEOZL_CODECS_FLOATMULT_ENCODE_KERNEL_H
#define GEOZL_CODECS_FLOATMULT_ENCODE_KERNEL_H

#include <stddef.h>
#include <stdint.h>

// Split a float stream against a float base into a mult stream and a ULP
// adjustment stream. mult = round(x / base) is stored via the integer-float
// latent, adj is the signed latent difference between x and mult*base, centered
// so that an exactly-on-grid value yields adj 0. primary and secondary each
// hold nb_elts elements of elt_width bytes. elt_width must be 4 or 8. base and
// inv_base must be a matched pair (inv_base == 1/base as computed by the
// caller) so the decoder reproduces mult*base bit for bit.
void floatmult_split(void *primary, void *secondary, const void *src,
                     size_t nb_elts, size_t elt_width, double base,
                     double inv_base);

#endif // GEOZL_CODECS_FLOATMULT_ENCODE_KERNEL_H
