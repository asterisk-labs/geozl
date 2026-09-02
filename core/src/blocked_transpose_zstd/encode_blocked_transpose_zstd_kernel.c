#include "encode_blocked_transpose_zstd_kernel.h"

#include <stdint.h>
#include <string.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define GEOZL_BLOCKED_NEON 1
#else
#define GEOZL_BLOCKED_NEON 0
#endif

static int host_is_little_endian(void) {
  const uint16_t one = 1;
  return *(const uint8_t *)&one == 1;
}

void blocked_transpose_zstd_shuffle(void *dst0, const void *src0,
                                    size_t nbElts, size_t eltWidth) {
  uint8_t *dst = (uint8_t *)dst0;
  const uint8_t *src = (const uint8_t *)src0;
  if (eltWidth == 1) {
    memmove(dst, src, nbElts);
    return;
  }

  const int little = host_is_little_endian();
  size_t i = 0;
#if GEOZL_BLOCKED_NEON
  if (little && eltWidth == 2) {
    for (; i + 16 <= nbElts; i += 16) {
      const uint8x16x2_t v = vld2q_u8(src + 2 * i);
      vst1q_u8(dst + i, v.val[0]);
      vst1q_u8(dst + nbElts + i, v.val[1]);
    }
  } else if (little && eltWidth == 4) {
    for (; i + 16 <= nbElts; i += 16) {
      const uint8x16x4_t v = vld4q_u8(src + 4 * i);
      vst1q_u8(dst + i, v.val[0]);
      vst1q_u8(dst + nbElts + i, v.val[1]);
      vst1q_u8(dst + 2 * nbElts + i, v.val[2]);
      vst1q_u8(dst + 3 * nbElts + i, v.val[3]);
    }
  }
#endif
  for (; i < nbElts; ++i) {
    for (size_t lane = 0; lane < eltWidth; ++lane) {
      const size_t nativeLane = little ? lane : eltWidth - 1 - lane;
      dst[lane * nbElts + i] = src[i * eltWidth + nativeLane];
    }
  }
}
