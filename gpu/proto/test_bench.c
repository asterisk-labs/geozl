#include "bench.h"

#include <stdio.h>
#include <string.h>

static int failures;
const char *gpu_backend_name = "test";

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);     \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

static void test_pfor_block_info(void) {
  uint8_t block[258];
  PforBlockInfo info;

  memset(block, 0, sizeof(block));
  CHECK(pfor_block_info(block, 2, 1, &info) == 0);
  CHECK(info.bytes == 2 && info.exceptions == 0);

  block[0] = 8;
  CHECK(pfor_block_info(block, sizeof(block), 1, &info) == 0);
  CHECK(info.bytes == sizeof(block));
  CHECK(pfor_block_info(block, sizeof(block) - 1, 1, &info) != 0);

  block[0] = 9;
  CHECK(pfor_block_info(block, sizeof(block), 1, &info) != 0);

  memset(block, 0, sizeof(block));
  block[0] = 1;
  block[1] = 1;
  block[2] = 7; // bitmap mode, seven high bits
  CHECK(pfor_block_info(block, 68, 1, &info) == 0);
  CHECK(info.bytes == 68 && info.exceptions == 1);
  CHECK(pfor_block_info(block, 67, 1, &info) != 0);

  block[2] = 0; // an exception must carry at least one high bit
  CHECK(pfor_block_info(block, sizeof(block), 1, &info) != 0);
  block[2] = 8; // low and high widths exceed uint8
  CHECK(pfor_block_info(block, sizeof(block), 1, &info) != 0);

  block[2] = 0x87; // list mode, one position byte and one data byte
  CHECK(pfor_block_info(block, 37, 1, &info) == 0);
  CHECK(info.bytes == 37 && info.exceptions == 1);
}

int main(void) {
  test_pfor_block_info();
  if (failures != 0)
    return 1;
  printf("test_bench: ok\n");
  return 0;
}
