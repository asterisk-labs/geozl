// Prefix sum scans, one per element width: dst[0]=src[0], dst[i]=dst[i-1]+src[i].
//
// Shared verbatim by the horizontal predictor decoders. Changes must keep all
// four widths in step; test/test_simd.c.
//
// On x86 both paths are compiled and the choice is made per call, because a
// wheel targets the base ISA and would otherwise ship without AVX2 at all.

#ifndef GEOZL_COMMON_SCAN_H
#define GEOZL_COMMON_SCAN_H

#include <stddef.h>
#include <stdint.h>

#include "common/simd.h"

#define GEOZL_SCAN_TAIL(T)                                                     \
  T acc = (i > 0) ? dst[i - 1] : 0;                                            \
  for (; i < n; ++i) {                                                         \
    acc = (T)(acc + src[i]);                                                   \
    dst[i] = acc;                                                              \
  }

#if GEOZL_SIMD_CAN_AVX2
GEOZL_TARGET_AVX2 static inline void geozl_scan8_avx2(uint8_t *dst, const uint8_t *src, size_t n) {
  size_t i = 0;
  __m256i carry = _mm256_setzero_si256();
  for (; i + 32 <= n; i += 32) {
    __m256i x = _mm256_loadu_si256((const __m256i *)(src + i));
    x = _mm256_add_epi8(x, _mm256_slli_si256(x, 1));
    x = _mm256_add_epi8(x, _mm256_slli_si256(x, 2));
    x = _mm256_add_epi8(x, _mm256_slli_si256(x, 4));
    x = _mm256_add_epi8(x, _mm256_slli_si256(x, 8));
    __m128i loTot =
        _mm_broadcastb_epi8(_mm_srli_si128(_mm256_castsi256_si128(x), 15));
    x = _mm256_add_epi8(
        x, _mm256_inserti128_si256(_mm256_setzero_si256(), loTot, 1));
    x = _mm256_add_epi8(x, carry);
    _mm256_storeu_si256((__m256i *)(dst + i), x);
    carry = _mm256_broadcastb_epi8(
        _mm_srli_si128(_mm256_extracti128_si256(x, 1), 15));
  }
  GEOZL_SCAN_TAIL(uint8_t)
}
#endif
#if GEOZL_SIMD_X86
static inline void geozl_scan8_sse2(uint8_t *dst, const uint8_t *src, size_t n) {
  size_t i = 0;
  __m128i carry = _mm_setzero_si128();
  for (; i + 16 <= n; i += 16) {
    __m128i x = _mm_loadu_si128((const __m128i *)(src + i));
    x = _mm_add_epi8(x, _mm_slli_si128(x, 1));
    x = _mm_add_epi8(x, _mm_slli_si128(x, 2));
    x = _mm_add_epi8(x, _mm_slli_si128(x, 4));
    x = _mm_add_epi8(x, _mm_slli_si128(x, 8));
    x = _mm_add_epi8(x, carry);
    _mm_storeu_si128((__m128i *)(dst + i), x);
    carry = _mm_set1_epi8((char)(_mm_extract_epi16(x, 7) >> 8));
  }
  GEOZL_SCAN_TAIL(uint8_t)
}
#elif GEOZL_SIMD_HAS_NEON
static inline void geozl_scan8_neon(uint8_t *dst, const uint8_t *src, size_t n) {
  size_t i = 0;
  const uint8x16_t zero = vdupq_n_u8(0);
  uint8x16_t carry = zero;
  for (; i + 16 <= n; i += 16) {
    uint8x16_t x = vld1q_u8(src + i);
    x = vaddq_u8(x, vextq_u8(zero, x, 15));
    x = vaddq_u8(x, vextq_u8(zero, x, 14));
    x = vaddq_u8(x, vextq_u8(zero, x, 12));
    x = vaddq_u8(x, vextq_u8(zero, x, 8));
    x = vaddq_u8(x, carry);
    vst1q_u8(dst + i, x);
    carry = vdupq_n_u8(vgetq_lane_u8(x, 15));
  }
  GEOZL_SCAN_TAIL(uint8_t)
}
#endif

static inline void geozl_scan8(uint8_t *dst, const uint8_t *src, size_t n) {
#if GEOZL_SIMD_CAN_AVX2
  if (geozl_simd_now() >= GEOZL_SIMD_AVX2) {
    geozl_scan8_avx2(dst, src, n);
    return;
  }
#endif
#if GEOZL_SIMD_X86
  geozl_scan8_sse2(dst, src, n);
#elif GEOZL_SIMD_HAS_NEON
  geozl_scan8_neon(dst, src, n);
#else
  size_t i = 0;
  GEOZL_SCAN_TAIL(uint8_t)
#endif
}

#if GEOZL_SIMD_CAN_AVX2
GEOZL_TARGET_AVX2 static inline void geozl_scan16_avx2(uint16_t *dst, const uint16_t *src, size_t n) {
  size_t i = 0;
  __m256i carry = _mm256_setzero_si256();
  for (; i + 16 <= n; i += 16) {
    __m256i x = _mm256_loadu_si256((const __m256i *)(src + i));
    x = _mm256_add_epi16(x, _mm256_slli_si256(x, 2));
    x = _mm256_add_epi16(x, _mm256_slli_si256(x, 4));
    x = _mm256_add_epi16(x, _mm256_slli_si256(x, 8));
    __m128i loTot =
        _mm_broadcastw_epi16(_mm_srli_si128(_mm256_castsi256_si128(x), 14));
    x = _mm256_add_epi16(
        x, _mm256_inserti128_si256(_mm256_setzero_si256(), loTot, 1));
    x = _mm256_add_epi16(x, carry);
    _mm256_storeu_si256((__m256i *)(dst + i), x);
    carry = _mm256_broadcastw_epi16(
        _mm_srli_si128(_mm256_extracti128_si256(x, 1), 14));
  }
  GEOZL_SCAN_TAIL(uint16_t)
}
#endif
#if GEOZL_SIMD_X86
static inline void geozl_scan16_sse2(uint16_t *dst, const uint16_t *src, size_t n) {
  size_t i = 0;
  __m128i carry = _mm_setzero_si128();
  for (; i + 8 <= n; i += 8) {
    __m128i x = _mm_loadu_si128((const __m128i *)(src + i));
    x = _mm_add_epi16(x, _mm_slli_si128(x, 2));
    x = _mm_add_epi16(x, _mm_slli_si128(x, 4));
    x = _mm_add_epi16(x, _mm_slli_si128(x, 8));
    x = _mm_add_epi16(x, carry);
    _mm_storeu_si128((__m128i *)(dst + i), x);
    carry = _mm_set1_epi16((short)_mm_extract_epi16(x, 7));
  }
  GEOZL_SCAN_TAIL(uint16_t)
}
#elif GEOZL_SIMD_HAS_NEON
static inline void geozl_scan16_neon(uint16_t *dst, const uint16_t *src, size_t n) {
  size_t i = 0;
  const uint16x8_t zero = vdupq_n_u16(0);
  uint16x8_t carry = zero;
  for (; i + 8 <= n; i += 8) {
    uint16x8_t x = vld1q_u16(src + i);
    x = vaddq_u16(x, vextq_u16(zero, x, 7));
    x = vaddq_u16(x, vextq_u16(zero, x, 6));
    x = vaddq_u16(x, vextq_u16(zero, x, 4));
    x = vaddq_u16(x, carry);
    vst1q_u16(dst + i, x);
    carry = vdupq_n_u16(vgetq_lane_u16(x, 7));
  }
  GEOZL_SCAN_TAIL(uint16_t)
}
#endif

static inline void geozl_scan16(uint16_t *dst, const uint16_t *src, size_t n) {
#if GEOZL_SIMD_CAN_AVX2
  if (geozl_simd_now() >= GEOZL_SIMD_AVX2) {
    geozl_scan16_avx2(dst, src, n);
    return;
  }
#endif
#if GEOZL_SIMD_X86
  geozl_scan16_sse2(dst, src, n);
#elif GEOZL_SIMD_HAS_NEON
  geozl_scan16_neon(dst, src, n);
#else
  size_t i = 0;
  GEOZL_SCAN_TAIL(uint16_t)
#endif
}

#if GEOZL_SIMD_CAN_AVX2
GEOZL_TARGET_AVX2 static inline void geozl_scan32_avx2(uint32_t *dst, const uint32_t *src, size_t n) {
  size_t i = 0;
  __m256i carry = _mm256_setzero_si256();
  for (; i + 8 <= n; i += 8) {
    __m256i x = _mm256_loadu_si256((const __m256i *)(src + i));
    x = _mm256_add_epi32(x, _mm256_slli_si256(x, 4));
    x = _mm256_add_epi32(x, _mm256_slli_si256(x, 8));
    __m128i loTot = _mm_shuffle_epi32(_mm256_castsi256_si128(x), 0xFF);
    x = _mm256_add_epi32(
        x, _mm256_inserti128_si256(_mm256_setzero_si256(), loTot, 1));
    x = _mm256_add_epi32(x, carry);
    _mm256_storeu_si256((__m256i *)(dst + i), x);
    carry = _mm256_broadcastd_epi32(
        _mm_shuffle_epi32(_mm256_extracti128_si256(x, 1), 0xFF));
  }
  GEOZL_SCAN_TAIL(uint32_t)
}
#endif
#if GEOZL_SIMD_X86
static inline void geozl_scan32_sse2(uint32_t *dst, const uint32_t *src, size_t n) {
  size_t i = 0;
  __m128i carry = _mm_setzero_si128();
  for (; i + 4 <= n; i += 4) {
    __m128i x = _mm_loadu_si128((const __m128i *)(src + i));
    x = _mm_add_epi32(x, _mm_slli_si128(x, 4));
    x = _mm_add_epi32(x, _mm_slli_si128(x, 8));
    x = _mm_add_epi32(x, carry);
    _mm_storeu_si128((__m128i *)(dst + i), x);
    carry = _mm_shuffle_epi32(x, _MM_SHUFFLE(3, 3, 3, 3));
  }
  GEOZL_SCAN_TAIL(uint32_t)
}
#elif GEOZL_SIMD_HAS_NEON
static inline void geozl_scan32_neon(uint32_t *dst, const uint32_t *src, size_t n) {
  size_t i = 0;
  const uint32x4_t zero = vdupq_n_u32(0);
  uint32x4_t carry = zero;
  for (; i + 4 <= n; i += 4) {
    uint32x4_t x = vld1q_u32(src + i);
    x = vaddq_u32(x, vextq_u32(zero, x, 3));
    x = vaddq_u32(x, vextq_u32(zero, x, 2));
    x = vaddq_u32(x, carry);
    vst1q_u32(dst + i, x);
    carry = vdupq_n_u32(vgetq_lane_u32(x, 3));
  }
  GEOZL_SCAN_TAIL(uint32_t)
}
#endif

static inline void geozl_scan32(uint32_t *dst, const uint32_t *src, size_t n) {
#if GEOZL_SIMD_CAN_AVX2
  if (geozl_simd_now() >= GEOZL_SIMD_AVX2) {
    geozl_scan32_avx2(dst, src, n);
    return;
  }
#endif
#if GEOZL_SIMD_X86
  geozl_scan32_sse2(dst, src, n);
#elif GEOZL_SIMD_HAS_NEON
  geozl_scan32_neon(dst, src, n);
#else
  size_t i = 0;
  GEOZL_SCAN_TAIL(uint32_t)
#endif
}

// 64 bit stays scalar: one add per element already saturates memory bandwidth,
// and the shuffles that carry across a vector cost more than they save here.
static inline void geozl_scan64(uint64_t *dst, const uint64_t *src, size_t n) {
  uint64_t acc = 0;
  for (size_t i = 0; i < n; ++i) {
    acc += src[i];
    dst[i] = acc;
  }
}

#endif // GEOZL_COMMON_SCAN_H
