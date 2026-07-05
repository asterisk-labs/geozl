#include "auto_delta_w.h"
#include "header_delta_w.h"
#include "encode_delta_w_kernel.h"
#include "decode_delta_w_kernel.h"

size_t delta_w_encode_auto(void* dst, uint8_t* header, size_t header_cap,
                            const void* src, size_t width, size_t nb_elts,
                            size_t elt_width)
{
    if (header_cap < DELTA_W_HEADER_SIZE)
        return 0;
    delta_w_encode(dst, src, width, nb_elts, elt_width);
    return delta_w_write_header(header, (uint32_t)width);
}

void delta_w_decode_auto(void* dst, const void* src, const uint8_t* header,
                          size_t header_size, size_t nb_elts, size_t elt_width)
{
    uint32_t width;
    if (!delta_w_read_header(header, header_size, &width))
        return;
    delta_w_decode(dst, src, width, nb_elts, elt_width);
}
