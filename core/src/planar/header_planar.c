#include "header_planar.h"

#include <string.h>

size_t planar_write_header(uint8_t* buf, uint32_t width)
{
    memcpy(buf, &width, sizeof(width));
    return PLANAR_HEADER_SIZE;
}

int planar_read_header(const uint8_t* buf, size_t size, uint32_t* width)
{
    if (size != PLANAR_HEADER_SIZE)
        return 0;
    memcpy(width, buf, sizeof(*width));
    return 1;
}
