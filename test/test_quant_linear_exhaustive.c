// The two branchless half conversions against the reference, over their whole
// domain. The first draft of ql_f32_to_half was wrong on two of the 2^32, both
// where the subnormal rounding carries into the smallest normal, and no sampled
// test would have found them.

#include "quant_linear/quant_linear_half.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int mismatch(const char *what, uint32_t bits, uint16_t want,
                    uint16_t got) {
  printf("  FAIL %s at 0x%08x: reference %04x, branchless %04x\n", what, bits,
         want, got);
  return 1;
}

static int all_int16(void) {
  puts("ql_i16_to_half over all 65536 int16");
  uint64_t bad = 0;
  for (int32_t v = -32768; v <= 32767; ++v) {
    const uint16_t want = quant_linear_float_to_half((float)v);
    const uint16_t got = ql_i16_to_half(v);
    if (want != got && bad++ < 8)
      mismatch("ql_i16_to_half", (uint32_t)v, want, got);
  }
  if (bad != 0)
    printf("  %llu differ\n", (unsigned long long)bad);
  return bad != 0;
}

static int all_float32(void) {
  puts("ql_f32_to_half over all 2^32 float32");
  uint64_t bad = 0;
  for (uint64_t u = 0; u <= 0xFFFFFFFFull; ++u) {
    const uint32_t bits = (uint32_t)u;
    float f;
    memcpy(&f, &bits, sizeof(f));
    const uint16_t want = quant_linear_float_to_half(f);
    const uint16_t got = ql_f32_to_half(f);
    if (want != got && bad++ < 8)
      mismatch("ql_f32_to_half", bits, want, got);
  }
  if (bad != 0)
    printf("  %llu differ\n", (unsigned long long)bad);
  return bad != 0;
}

int main(void) {
  int bad = 0;
  bad |= all_int16();
  bad |= all_float32();
  if (bad) {
    puts("failed");
    return 1;
  }
  puts("all passed");
  return 0;
}
