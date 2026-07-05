#include "header_delta_w.h"

#include <string.h>

size_t delta_w_write_header(uint8_t* buf, uint32_t width)
{
    memcpy(buf, &width, sizeof(width));
    return DELTA_W_HEADER_SIZE;
}

int delta_w_read_header(const uint8_t* buf, size_t size, uint32_t* width)
{
    if (size != DELTA_W_HEADER_SIZE)
        return 0;
    memcpy(width, buf, sizeof(*width));
    return 1;
}
