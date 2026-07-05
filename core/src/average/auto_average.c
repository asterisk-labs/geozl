#include "auto_average.h"
#include "header_average.h"
#include "encode_average_kernel.h"
#include "decode_average_kernel.h"

size_t average_encode_auto(void* dst, uint8_t* header, size_t header_cap,
                            const void* src, size_t width, size_t nb_elts,
                            size_t elt_width)
{
    if (header_cap < AVERAGE_HEADER_SIZE)
        return 0;
    average_encode(dst, src, width, nb_elts, elt_width);
    return average_write_header(header, (uint32_t)width);
}

void average_decode_auto(void* dst, const void* src, const uint8_t* header,
                          size_t header_size, size_t nb_elts, size_t elt_width)
{
    uint32_t width;
    if (!average_read_header(header, header_size, &width))
        return;
    average_decode(dst, src, width, nb_elts, elt_width);
}
