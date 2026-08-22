#include "geozl/coeffs.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define MAX_VECS GEOZL_COEFFS_MAX_VECS
#define MAX_VALUES 2500

static int32_t values[MAX_VALUES];
static uint32_t counts[MAX_VECS];

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  // Exercise foreign data and inputs with geozl's magic.
  static uint8_t ours[GEOZL_COEFFS_MAX_BYTES + 8];

  for (int mine = 0; mine < 2; ++mine) {
    const uint8_t *p = data;
    size_t n = size;
    if (mine) {
      if (size < 4 || size > sizeof ours)
        continue;
      memcpy(ours, data, size);
      memcpy(ours, "GZC1", 4);
      if (size >= 14 && (size - 10) % 4 == 0) {
        uint32_t valueCount = (uint32_t)((size - 10) / 4);
        ours[4] = 1;
        ours[5] = 1;
        ours[6] = (uint8_t)valueCount;
        ours[7] = (uint8_t)(valueCount >> 8);
        ours[8] = (uint8_t)(valueCount >> 16);
        ours[9] = (uint8_t)(valueCount >> 24);
      }
      p = ours;
      n = size;
    }

    size_t vecs = 0, vals = 0;
    int rc = geozl_coeffs_parse(p, n, NULL, 0, NULL, 0, &vecs, &vals);
    if (rc != 0)
      continue;

    if (vecs == 0 || vecs > MAX_VECS || vals == 0 || vals > MAX_VALUES)
      __builtin_trap();
    if (geozl_coeffs_parse(p, n, values, vals, counts, vecs, NULL, NULL) != 0)
      __builtin_trap();

    size_t sum = 0;
    for (size_t i = 0; i < vecs; ++i) {
      if (counts[i] == 0)
        __builtin_trap();
      sum += counts[i];
    }
    if (sum != vals)
      __builtin_trap();

    if (geozl_coeffs_parse(p, n, values, vals - 1, NULL, 0, NULL, NULL) != -2)
      __builtin_trap();
    if (geozl_coeffs_parse(p, n, NULL, 0, counts, vecs - 1, NULL, NULL) != -2)
      __builtin_trap();
  }
  return 0;
}
