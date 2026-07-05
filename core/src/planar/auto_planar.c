#include "auto_planar.h"
#include "header_planar.h"
#include "encode_planar_kernel.h"
#include "decode_planar_kernel.h"

size_t planar_encode_auto(void* dst, uint8_t* header, size_t header_cap,
                            const void* src, size_t width, size_t nb_elts,
                            size_t elt_width)
{
    if (header_cap < PLANAR_HEADER_SIZE)
        return 0;
    planar_encode(dst, src, width, nb_elts, elt_width);
    return planar_write_header(header, (uint32_t)width);
}

void planar_decode_auto(void* dst, const void* src, const uint8_t* header,
                          size_t header_size, size_t nb_elts, size_t elt_width)
{
    uint32_t width;
    if (!planar_read_header(header, header_size, &width))
        return;
    planar_decode(dst, src, width, nb_elts, elt_width);
}
