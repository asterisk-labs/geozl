#include "transpose_pivco/decode_transpose_pivco_kernel.h"
#include "transpose_pivco/encode_transpose_pivco_kernel.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int failures;

#define CHECK(c)                                                               \
  do {                                                                         \
    if (!(c)) {                                                                \
      fprintf(stderr, "test_transpose_pivco:%d: %s\n", __LINE__, #c);          \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

static void check_width(size_t width) {
  enum { count = 257 };
  uint8_t source[count * 8];
  uint8_t lanes[count * 8];
  uint8_t decoded[count * 8];
  for (size_t i = 0; i < count * width; ++i)
    source[i] = (uint8_t)(i * 131u + i / 7u);
  memset(lanes, 0, sizeof(lanes));
  memset(decoded, 0, sizeof(decoded));
  CHECK(transpose_pivco_lanes(lanes, source, count, width) == 0);
  CHECK(transpose_pivco_unlanes(decoded, lanes, count, width) == 0);
  CHECK(memcmp(source, decoded, count * width) == 0);
}

int main(void) {
  check_width(1);
  check_width(2);
  check_width(4);
  check_width(8);

  uint8_t byte = 0;
  CHECK(transpose_pivco_lanes(NULL, &byte, 1, 1) != 0);
  CHECK(transpose_pivco_lanes(&byte, NULL, 1, 1) != 0);
  CHECK(transpose_pivco_lanes(&byte, &byte, 0, 1) != 0);
  CHECK(transpose_pivco_lanes(&byte, &byte, 1, 3) != 0);
  CHECK(transpose_pivco_unlanes(NULL, &byte, 1, 1) != 0);
  CHECK(transpose_pivco_unlanes(&byte, NULL, 1, 1) != 0);

  if (failures != 0) {
    fprintf(stderr, "test_transpose_pivco: %d failures\n", failures);
    return 1;
  }
  puts("test_transpose_pivco: ok");
  return 0;
}
