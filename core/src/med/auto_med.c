#include "auto_med.h"
#include "header_med.h"
#include "encode_med_kernel.h"
#include "decode_med_kernel.h"

size_t med_encode_auto(void* dst, uint8_t* header, size_t header_cap,
                            const void* src, size_t width, size_t nb_elts,
                            size_t elt_width)
{
    if (header_cap < MED_HEADER_SIZE)
        return 0;
    med_encode(dst, src, width, nb_elts, elt_width);
    return med_write_header(header, (uint32_t)width);
}

void med_decode_auto(void* dst, const void* src, const uint8_t* header,
                          size_t header_size, size_t nb_elts, size_t elt_width)
{
    uint32_t width;
    if (!med_read_header(header, header_size, &width))
        return;
    med_decode(dst, src, width, nb_elts, elt_width);
}
