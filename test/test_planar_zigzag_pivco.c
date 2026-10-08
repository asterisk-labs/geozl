#include "planar_zigzag/encode_planar_zigzag_kernel.h"
#include "planar_zigzag_pivco/decode_planar_zigzag_pivco_kernel.h"
#include "planar_zigzag_pivco/encode_planar_zigzag_pivco_kernel.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);               \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

#define MAXN 4096

static uint64_t source[MAXN], transformed[MAXN], decoded[MAXN];
static uint8_t lanes[MAXN * 8];

static uint64_t rng = UINT64_C(0x243f6a8885a308d3);
static uint64_t rnd(void) {
  rng ^= rng << 13;
  rng ^= rng >> 7;
  rng ^= rng << 17;
  return rng;
}

static void put(size_t i, size_t width, uint64_t value) {
  switch (width) {
  case 1:
    ((uint8_t *)source)[i] = (uint8_t)value;
    break;
  case 2:
    ((uint16_t *)source)[i] = (uint16_t)value;
    break;
  case 4:
    ((uint32_t *)source)[i] = (uint32_t)value;
    break;
  default:
    source[i] = value;
    break;
  }
}

// The lanes hold exactly the bytes of planar_zigzag's residuals, and joining
// them gives the source back.
static void trip(size_t n, size_t rowWidth, uint32_t planes, size_t eltWidth,
                 int smooth) {
  CHECK(n <= MAXN);
  for (size_t i = 0; i < n; ++i)
    put(i, eltWidth,
        smooth ? (uint64_t)((i % rowWidth) * 3 + (i / rowWidth) * 5 + (rnd() & 3))
               : rnd() ^ (i * UINT64_C(0x9e3779b97f4a7c15)));

  const size_t planeElts = n / planes;
  for (uint32_t plane = 0; plane < planes; ++plane) {
    const size_t at = (size_t)plane * planeElts * eltWidth;
    CHECK(planar_zigzag_encode((uint8_t *)transformed + at,
                               (const uint8_t *)source + at, rowWidth,
                               planeElts, eltWidth) == 0);
  }

  CHECK(planar_zigzag_pivco_lanes(lanes, source, rowWidth, n, eltWidth,
                                  planes) == 0);
  const uint8_t *t = (const uint8_t *)transformed;
  int same = 1;
  for (size_t i = 0; i < n; ++i)
    for (size_t k = 0; k < eltWidth; ++k)
      same &= lanes[k * n + i] == t[i * eltWidth + k];
  CHECK(same);

  memset(decoded, 0xAB, sizeof(decoded));
  CHECK(planar_zigzag_pivco_unlanes(decoded, lanes, rowWidth, n, eltWidth,
                                    planes) == 0);
  CHECK(memcmp(decoded, source, n * eltWidth) == 0);
}

int main(void) {
  const size_t widths[] = {1, 2, 4, 8};
  for (size_t i = 0; i < sizeof(widths) / sizeof(widths[0]); ++i) {
    const size_t elt = widths[i];
    for (int smooth = 0; smooth < 2; ++smooth) {
      trip(13 * 17, 17, 1, elt, smooth);
      trip(2 * 13 * 17, 17, 2, elt, smooth);
      trip(3 * 17 * 19, 19, 3, elt, smooth);
      trip(2 * 257, 257, 1, elt, smooth);
      trip(1024, 1, 1, elt, smooth);
      trip(1, 1, 1, elt, smooth);
    }
  }

  // Geometry the kernels refuse, leaving the buffers untouched.
  CHECK(planar_zigzag_pivco_lanes(lanes, source, 0, 256, 2, 1) != 0);
  CHECK(planar_zigzag_pivco_lanes(lanes, source, 16, 255, 2, 1) != 0);
  CHECK(planar_zigzag_pivco_lanes(lanes, source, 16, 256, 3, 1) != 0);
  CHECK(planar_zigzag_pivco_lanes(lanes, source, 16, 256, 2, 0) != 0);
  CHECK(planar_zigzag_pivco_lanes(lanes, source, 16, 0, 2, 1) != 0);
  CHECK(planar_zigzag_pivco_unlanes(decoded, lanes, 0, 256, 2, 1) != 0);
  CHECK(planar_zigzag_pivco_unlanes(decoded, lanes, 16, 256, 3, 1) != 0);
  CHECK(planar_zigzag_pivco_unlanes(decoded, lanes, 16, 256, 2, 0) != 0);
  CHECK(planar_zigzag_pivco_unlanes(decoded, lanes, 16, 255, 2, 1) != 0);

  if (failures) {
    printf("test_planar_zigzag_pivco: %d failures\n", failures);
    return 1;
  }
  printf("test_planar_zigzag_pivco: ok\n");
  return 0;
}
