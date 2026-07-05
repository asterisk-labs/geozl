#ifndef GEOZL_CODECS_DELTA_W_HEADER_H
#define GEOZL_CODECS_DELTA_W_HEADER_H

#include <stddef.h>
#include <stdint.h>

// Header is a single little endian uint32, the row width in samples.
#define DELTA_W_HEADER_SIZE 4

size_t delta_w_write_header(uint8_t* buf, uint32_t width);
int delta_w_read_header(const uint8_t* buf, size_t size, uint32_t* width);

#endif // GEOZL_CODECS_DELTA_W_HEADER_H
