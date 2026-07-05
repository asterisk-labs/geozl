#ifndef GEOZL_CODECS_WP_STATIC_HEADER_H
#define GEOZL_CODECS_WP_STATIC_HEADER_H

#include <stddef.h>
#include <stdint.h>

// The wp_static codec header, little endian: uint32 width, uint8 shift, then
// four int16 coefficients {cN, cNW, cNE, cNN}. This is the one definition of the
// layout, shared by the encode and decode paths and the OpenZL bindings.
#define WP_STATIC_HEADER_SIZE (4 + 1 + 4 * 2)

// Serialize the kernel into buf, which must hold WP_STATIC_HEADER_SIZE bytes.
// Returns the number of bytes written.
size_t wp_static_write_header(uint8_t* buf, uint32_t width,
                              const int16_t coeffs[4], uint8_t shift);

// Parse a header. Returns 1 on success, 0 if the size is wrong.
int wp_static_read_header(const uint8_t* buf, size_t size, uint32_t* width,
                          int16_t coeffs[4], uint8_t* shift);

#endif // GEOZL_CODECS_WP_STATIC_HEADER_H
