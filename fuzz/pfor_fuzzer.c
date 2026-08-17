// The first byte selects the element width and round-trip or decode-only mode.

#include "pfor/decode_pfor_kernel.h"
#include "pfor/encode_pfor_kernel.h"
#include "pfor/pfor_check.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ELTS 4096

// Keep the arbitrary input aligned for the encoder.
static uint64_t aligned[MAX_ELTS];

static const size_t kWidths[4] = {1, 2, 4, 8};

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  if (size < 2)
    return 0;
  const size_t eltWidth = kWidths[data[0] & 3u];
  const int roundTrip = (data[0] >> 2) & 1;
  ++data;
  --size;

  if (roundTrip) {
    size_t nbElts = size / eltWidth;
    if (nbElts == 0)
      return 0;
    if (nbElts > MAX_ELTS)
      nbElts = MAX_ELTS;
    if (nbElts * eltWidth > sizeof(aligned))
      nbElts = sizeof(aligned) / eltWidth;

    const size_t bound = pfor_bound(nbElts, eltWidth);
    if (bound == 0)
      return 0;
    uint8_t *packed = (uint8_t *)malloc(bound);
    uint8_t *back = (uint8_t *)malloc(nbElts * eltWidth);
    if (packed == NULL || back == NULL) {
      free(packed);
      free(back);
      return 0;
    }
    memcpy(aligned, data, nbElts * eltWidth);
    size_t written = 0;
    if (pfor_encode(packed, bound, &written, aligned, nbElts, eltWidth) == 0) {
      if (pfor_decode(back, nbElts, eltWidth, packed, written) != 0)
        abort();
      if (memcmp(back, aligned, nbElts * eltWidth) != 0)
        abort();
    }
    free(packed);
    free(back);
    return 0;
  }

  // Let the element count disagree with the encoded stream.
  size_t nbElts = ((size_t)data[0] << 8 | (size > 1 ? data[1] : 0)) + 1;
  if (nbElts > MAX_ELTS)
    nbElts = MAX_ELTS;
  uint8_t *out = (uint8_t *)malloc(nbElts * eltWidth);
  if (out == NULL)
    return 0;
  (void)pfor_decode(out, nbElts, eltWidth, data, size);
  free(out);
  return 0;
}
