// The one place the wp_static header layout is written and read, so no reader
// can drift from the writer.

#include "header_wp_static.h"

#include <string.h>

size_t wp_static_write_header(uint8_t* buf, uint32_t width,
                              const int16_t coeffs[4], uint8_t shift)
{
    memcpy(buf, &width, sizeof(width));
    buf[4] = shift;
    memcpy(buf + 5, coeffs, 4 * sizeof(int16_t));
    return WP_STATIC_HEADER_SIZE;
}

int wp_static_read_header(const uint8_t* buf, size_t size, uint32_t* width,
                          int16_t coeffs[4], uint8_t* shift)
{
    if (size != WP_STATIC_HEADER_SIZE)
        return 0;
    memcpy(width, buf, sizeof(*width));
    *shift = buf[4];
    memcpy(coeffs, buf + 5, 4 * sizeof(int16_t));
    return 1;
}
