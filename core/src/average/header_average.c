#include "header_average.h"

#include <string.h>

size_t average_write_header(uint8_t* buf, uint32_t width)
{
    memcpy(buf, &width, sizeof(width));
    return AVERAGE_HEADER_SIZE;
}

int average_read_header(const uint8_t* buf, size_t size, uint32_t* width)
{
    if (size != AVERAGE_HEADER_SIZE)
        return 0;
    memcpy(width, buf, sizeof(*width));
    return 1;
}
