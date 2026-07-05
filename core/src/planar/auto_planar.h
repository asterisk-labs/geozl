#ifndef GEOZL_CODECS_PLANAR_AUTO_H
#define GEOZL_CODECS_PLANAR_AUTO_H

#include <stddef.h>
#include <stdint.h>

// Returns the header size, or 0 if header_cap is too small.
size_t planar_encode_auto(void* dst, uint8_t* header, size_t header_cap,
                            const void* src, size_t width, size_t nb_elts,
                            size_t elt_width);

void planar_decode_auto(void* dst, const void* src, const uint8_t* header,
                          size_t header_size, size_t nb_elts, size_t elt_width);

#endif // GEOZL_CODECS_PLANAR_AUTO_H
