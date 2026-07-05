#ifndef GEOZL_CODECS_WP_STATIC_AUTO_H
#define GEOZL_CODECS_WP_STATIC_AUTO_H

#include <stddef.h>
#include <stdint.h>

// Estimate the kernel from the tile, write the header, and encode. Returns the
// header size, or 0 if header_cap is too small.
size_t wp_static_encode_auto(void* dst, uint8_t* header, size_t header_cap,
                             const void* src, size_t width, size_t nb_elts,
                             size_t elt_width);

// Read the header and decode.
void wp_static_decode_auto(void* dst, const void* src, const uint8_t* header,
                           size_t header_size, size_t nb_elts, size_t elt_width);

#endif // GEOZL_CODECS_WP_STATIC_AUTO_H
