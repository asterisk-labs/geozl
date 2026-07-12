#ifndef GEOZL_CODECS_FLOATQUANT_ENCODE_KERNEL_H
#define GEOZL_CODECS_FLOATQUANT_ENCODE_KERNEL_H

#include <stddef.h>
#include <stdint.h>

// Split a float stream into a high part and the low k mantissa bits, working
// on the order preserving unsigned image of each float. primary holds the
// value with its low k bits removed, secondary holds those k bits, flipped for
// negative floats so an exactly quantized value always yields secondary 0.
// Both streams hold nb_elts elements of elt_width bytes. elt_width must be 4 or
// 8. k must be in 1..=PRECISION_BITS (23 for f32, 52 for f64).
void floatquant_split(void *primary, void *secondary, const void *src,
                      size_t nb_elts, size_t elt_width, unsigned k);

#endif // GEOZL_CODECS_FLOATQUANT_ENCODE_KERNEL_H
