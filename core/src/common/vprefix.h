// The in-register prefix sum, one copy per lane width and instruction set.
//
// Two callers need it: the plain scans in scan.h and the fused planar rows in
// decode_planar_kernel.c, which differ only in how they build the vector they
// hand over. The ladder itself and the carry broadcast were written out twice
// before, which is two places to get a cross-lane fixup wrong.
//
// prefix() leaves lane k holding the sum of lanes 0..k. last() broadcasts the
// top lane, which is the running total to carry into the next block.

#ifndef GEOZL_COMMON_VPREFIX_H
#define GEOZL_COMMON_VPREFIX_H

#include "common/simd.h"

// AVX2 shifts each 128-bit half on its own, so after the ladder the low half's
// total has to be folded into the high half by hand. That fold is the one part
// of this that is easy to get wrong, which is the reason the file exists.
#if GEOZL_SIMD_CAN_AVX2

GEOZL_TARGET_AVX2 static inline __m256i geozl_vprefix8_avx2(__m256i x) {
  x = _mm256_add_epi8(x, _mm256_slli_si256(x, 1));
  x = _mm256_add_epi8(x, _mm256_slli_si256(x, 2));
  x = _mm256_add_epi8(x, _mm256_slli_si256(x, 4));
  x = _mm256_add_epi8(x, _mm256_slli_si256(x, 8));
  __m128i loTot =
      _mm_broadcastb_epi8(_mm_srli_si128(_mm256_castsi256_si128(x), 15));
  return _mm256_add_epi8(
      x, _mm256_inserti128_si256(_mm256_setzero_si256(), loTot, 1));
}

GEOZL_TARGET_AVX2 static inline __m256i geozl_vlast8_avx2(__m256i x) {
  return _mm256_broadcastb_epi8(
      _mm_srli_si128(_mm256_extracti128_si256(x, 1), 15));
}

GEOZL_TARGET_AVX2 static inline __m256i geozl_vprefix16_avx2(__m256i x) {
  x = _mm256_add_epi16(x, _mm256_slli_si256(x, 2));
  x = _mm256_add_epi16(x, _mm256_slli_si256(x, 4));
  x = _mm256_add_epi16(x, _mm256_slli_si256(x, 8));
  __m128i loTot =
      _mm_broadcastw_epi16(_mm_srli_si128(_mm256_castsi256_si128(x), 14));
  return _mm256_add_epi16(
      x, _mm256_inserti128_si256(_mm256_setzero_si256(), loTot, 1));
}

GEOZL_TARGET_AVX2 static inline __m256i geozl_vlast16_avx2(__m256i x) {
  return _mm256_broadcastw_epi16(
      _mm_srli_si128(_mm256_extracti128_si256(x, 1), 14));
}

GEOZL_TARGET_AVX2 static inline __m256i geozl_vprefix32_avx2(__m256i x) {
  x = _mm256_add_epi32(x, _mm256_slli_si256(x, 4));
  x = _mm256_add_epi32(x, _mm256_slli_si256(x, 8));
  __m128i loTot = _mm_shuffle_epi32(_mm256_castsi256_si128(x), 0xFF);
  return _mm256_add_epi32(
      x, _mm256_inserti128_si256(_mm256_setzero_si256(), loTot, 1));
}

GEOZL_TARGET_AVX2 static inline __m256i geozl_vlast32_avx2(__m256i x) {
  return _mm256_broadcastd_epi32(
      _mm_shuffle_epi32(_mm256_extracti128_si256(x, 1), 0xFF));
}

#endif // GEOZL_SIMD_CAN_AVX2

#if GEOZL_SIMD_X86

static inline __m128i geozl_vprefix8_sse2(__m128i x) {
  x = _mm_add_epi8(x, _mm_slli_si128(x, 1));
  x = _mm_add_epi8(x, _mm_slli_si128(x, 2));
  x = _mm_add_epi8(x, _mm_slli_si128(x, 4));
  x = _mm_add_epi8(x, _mm_slli_si128(x, 8));
  return x;
}

static inline __m128i geozl_vlast8_sse2(__m128i x) {
  return _mm_set1_epi8((char)(_mm_extract_epi16(x, 7) >> 8));
}

static inline __m128i geozl_vprefix16_sse2(__m128i x) {
  x = _mm_add_epi16(x, _mm_slli_si128(x, 2));
  x = _mm_add_epi16(x, _mm_slli_si128(x, 4));
  x = _mm_add_epi16(x, _mm_slli_si128(x, 8));
  return x;
}

static inline __m128i geozl_vlast16_sse2(__m128i x) {
  return _mm_set1_epi16((short)_mm_extract_epi16(x, 7));
}

static inline __m128i geozl_vprefix32_sse2(__m128i x) {
  x = _mm_add_epi32(x, _mm_slli_si128(x, 4));
  x = _mm_add_epi32(x, _mm_slli_si128(x, 8));
  return x;
}

static inline __m128i geozl_vlast32_sse2(__m128i x) {
  return _mm_shuffle_epi32(x, _MM_SHUFFLE(3, 3, 3, 3));
}

#elif GEOZL_SIMD_HAS_NEON

static inline uint8x16_t geozl_vprefix8_neon(uint8x16_t x) {
  const uint8x16_t zero = vdupq_n_u8(0);
  x = vaddq_u8(x, vextq_u8(zero, x, 15));
  x = vaddq_u8(x, vextq_u8(zero, x, 14));
  x = vaddq_u8(x, vextq_u8(zero, x, 12));
  x = vaddq_u8(x, vextq_u8(zero, x, 8));
  return x;
}

static inline uint8x16_t geozl_vlast8_neon(uint8x16_t x) {
  return vdupq_n_u8(vgetq_lane_u8(x, 15));
}

static inline uint16x8_t geozl_vprefix16_neon(uint16x8_t x) {
  const uint16x8_t zero = vdupq_n_u16(0);
  x = vaddq_u16(x, vextq_u16(zero, x, 7));
  x = vaddq_u16(x, vextq_u16(zero, x, 6));
  x = vaddq_u16(x, vextq_u16(zero, x, 4));
  return x;
}

static inline uint16x8_t geozl_vlast16_neon(uint16x8_t x) {
  return vdupq_n_u16(vgetq_lane_u16(x, 7));
}

static inline uint32x4_t geozl_vprefix32_neon(uint32x4_t x) {
  const uint32x4_t zero = vdupq_n_u32(0);
  x = vaddq_u32(x, vextq_u32(zero, x, 3));
  x = vaddq_u32(x, vextq_u32(zero, x, 2));
  return x;
}

static inline uint32x4_t geozl_vlast32_neon(uint32x4_t x) {
  return vdupq_n_u32(vgetq_lane_u32(x, 3));
}

#endif

#endif // GEOZL_COMMON_VPREFIX_H
