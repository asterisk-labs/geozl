// encode_auto trains, writes the header and encodes. decode_auto reads the
// header and decodes. The header layout itself lives in header_wp_static.

#include "auto_wp_static.h"
#include "header_wp_static.h"
#include "train_wp_static.h"
#include "encode_wp_static_kernel.h"
#include "decode_wp_static_kernel.h"

size_t wp_static_encode_auto(void* dst, uint8_t* header, size_t header_cap,
                             const void* src, size_t width, size_t nb_elts,
                             size_t elt_width)
{
    if (header_cap < WP_STATIC_HEADER_SIZE)
        return 0;
    int16_t coeffs[4];
    uint8_t shift;
    wp_static_train(coeffs, &shift, src, width, nb_elts, elt_width);
    wp_static_encode(dst, src, width, nb_elts, elt_width, coeffs, shift);
    return wp_static_write_header(header, (uint32_t)width, coeffs, shift);
}

void wp_static_decode_auto(void* dst, const void* src, const uint8_t* header,
                           size_t header_size, size_t nb_elts, size_t elt_width)
{
    uint32_t width;
    int16_t coeffs[4];
    uint8_t shift;
    if (!wp_static_read_header(header, header_size, &width, coeffs, &shift))
        return;
    wp_static_decode(dst, src, width, nb_elts, elt_width, coeffs, shift);
}
