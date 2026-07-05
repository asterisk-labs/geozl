#include "header_delta_n.h"

#include <string.h>

size_t delta_n_write_header(uint8_t* buf, uint32_t width)
{
    memcpy(buf, &width, sizeof(width));
    return DELTA_N_HEADER_SIZE;
}

int delta_n_read_header(const uint8_t* buf, size_t size, uint32_t* width)
{
    if (size != DELTA_N_HEADER_SIZE)
        return 0;
    memcpy(width, buf, sizeof(*width));
    return 1;
}
