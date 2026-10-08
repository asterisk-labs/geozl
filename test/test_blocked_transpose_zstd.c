#include "blocked_transpose_zstd/decode_blocked_transpose_zstd_kernel.h"

#include <assert.h>
#include <stdint.h>
#include <string.h>

static void make_lanes(uint8_t *dst, const uint8_t *src, size_t n,
                       size_t width) {
  const uint16_t one = 1;
  const int little = *(const uint8_t *)&one == 1;
  for (size_t i = 0; i < n; ++i) {
    for (size_t lane = 0; lane < width; ++lane) {
      const size_t native = little ? lane : width - 1 - lane;
      dst[lane * n + i] = src[i * width + native];
    }
  }
}

int main(void) {
  uint8_t src[8 * 67];
  uint8_t shuffled[sizeof(src)];
  uint8_t back[sizeof(src)];
  for (size_t i = 0; i < sizeof(src); ++i)
    src[i] = (uint8_t)(i * 29u + i / 7u);

  const size_t widths[] = {1, 2, 4, 8};
  for (size_t w = 0; w < sizeof(widths) / sizeof(widths[0]); ++w) {
    const size_t width = widths[w];
    const size_t n = sizeof(src) / width;
    memset(shuffled, 0, sizeof(shuffled));
    memset(back, 0, sizeof(back));
    make_lanes(shuffled, src, n, width);
    blocked_transpose_zstd_unshuffle(back, shuffled, n, width);
    assert(memcmp(src, back, n * width) == 0);
  }
  return 0;
}
