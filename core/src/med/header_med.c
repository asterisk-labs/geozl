#include "header_med.h"

#include <string.h>

size_t med_write_header(uint8_t* buf, uint32_t width)
{
    memcpy(buf, &width, sizeof(width));
    return MED_HEADER_SIZE;
}

int med_read_header(const uint8_t* buf, size_t size, uint32_t* width)
{
    if (size != MED_HEADER_SIZE)
        return 0;
    memcpy(width, buf, sizeof(*width));
    return 1;
}
