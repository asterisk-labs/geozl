// Inverse planar predictor, W + N - NW. The predictor is linear, so decoding a
// row is a prefix sum. Row zero is a plain prefix sum of the residual. Each
// later row folds the row above into the residual, c[i] = res[i] + N[i] -
// NW[i], and prefix sums it in the same pass, so the W chain is resolved
// without a separate fold. Row major, @width samples per row. @dst may alias
// @src.

#include "decode_planar_kernel.h"

#include "common/raster.h" // geozl_row_width

#include <stdint.h>

#include "common/scan.h"

#define PLANAR_AVX2 GEOZL_SCAN_AVX2
#define PLANAR_SSE2 GEOZL_SCAN_SSE2
#define PLANAR_NEON GEOZL_SCAN_NEON
#define scan8 geozl_scan8
#define scan16 geozl_scan16
#define scan32 geozl_scan32
#define scan64 geozl_scan64

// Interior row, fused. c[i] = res[i] + above[i] - above[i-1] is built in
// register and prefix summed in the same pass. The first sample has no left
// neighbour and no above left, so above[-1] is taken as zero. above is the
// row already reconstructed above this one.

static void frow8(uint8_t *d, const uint8_t *res, const uint8_t *above,
                  size_t n) {
  uint8_t acc = (uint8_t)(res[0] + above[0]);
  d[0] = acc;
  size_t i = 1;
#if PLANAR_AVX2
  __m256i carry = _mm256_set1_epi8((char)acc);
  for (; i + 32 <= n; i += 32) {
    __m256i c = _mm256_sub_epi8(
        _mm256_add_epi8(_mm256_loadu_si256((const __m256i *)(res + i)),
                        _mm256_loadu_si256((const __m256i *)(above + i))),
        _mm256_loadu_si256((const __m256i *)(above + i - 1)));
    c = _mm256_add_epi8(c, _mm256_slli_si256(c, 1));
    c = _mm256_add_epi8(c, _mm256_slli_si256(c, 2));
    c = _mm256_add_epi8(c, _mm256_slli_si256(c, 4));
    c = _mm256_add_epi8(c, _mm256_slli_si256(c, 8));
    __m128i loTot =
        _mm_broadcastb_epi8(_mm_srli_si128(_mm256_castsi256_si128(c), 15));
    c = _mm256_add_epi8(
        c, _mm256_inserti128_si256(_mm256_setzero_si256(), loTot, 1));
    c = _mm256_add_epi8(c, carry);
    _mm256_storeu_si256((__m256i *)(d + i), c);
    carry = _mm256_broadcastb_epi8(
        _mm_srli_si128(_mm256_extracti128_si256(c, 1), 15));
  }
#elif PLANAR_SSE2
  __m128i carry = _mm_set1_epi8((char)acc);
  for (; i + 16 <= n; i += 16) {
    __m128i c = _mm_sub_epi8(
        _mm_add_epi8(_mm_loadu_si128((const __m128i *)(res + i)),
                     _mm_loadu_si128((const __m128i *)(above + i))),
        _mm_loadu_si128((const __m128i *)(above + i - 1)));
    c = _mm_add_epi8(c, _mm_slli_si128(c, 1));
    c = _mm_add_epi8(c, _mm_slli_si128(c, 2));
    c = _mm_add_epi8(c, _mm_slli_si128(c, 4));
    c = _mm_add_epi8(c, _mm_slli_si128(c, 8));
    c = _mm_add_epi8(c, carry);
    _mm_storeu_si128((__m128i *)(d + i), c);
    carry = _mm_set1_epi8((char)(_mm_extract_epi16(c, 7) >> 8));
  }
#elif PLANAR_NEON
  const uint8x16_t zero = vdupq_n_u8(0);
  uint8x16_t carry = vdupq_n_u8(acc);
  for (; i + 16 <= n; i += 16) {
    uint8x16_t c = vsubq_u8(vaddq_u8(vld1q_u8(res + i), vld1q_u8(above + i)),
                            vld1q_u8(above + i - 1));
    c = vaddq_u8(c, vextq_u8(zero, c, 15));
    c = vaddq_u8(c, vextq_u8(zero, c, 14));
    c = vaddq_u8(c, vextq_u8(zero, c, 12));
    c = vaddq_u8(c, vextq_u8(zero, c, 8));
    c = vaddq_u8(c, carry);
    vst1q_u8(d + i, c);
    carry = vdupq_n_u8(vgetq_lane_u8(c, 15));
  }
#endif
  acc = d[i - 1];
  for (; i < n; ++i) {
    acc = (uint8_t)(acc + (uint8_t)(res[i] + above[i] - above[i - 1]));
    d[i] = acc;
  }
}

static void frow16(uint16_t *d, const uint16_t *res, const uint16_t *above,
                   size_t n) {
  uint16_t acc = (uint16_t)(res[0] + above[0]);
  d[0] = acc;
  size_t i = 1;
#if PLANAR_AVX2
  __m256i carry = _mm256_set1_epi16((short)acc);
  for (; i + 16 <= n; i += 16) {
    __m256i c = _mm256_sub_epi16(
        _mm256_add_epi16(_mm256_loadu_si256((const __m256i *)(res + i)),
                         _mm256_loadu_si256((const __m256i *)(above + i))),
        _mm256_loadu_si256((const __m256i *)(above + i - 1)));
    c = _mm256_add_epi16(c, _mm256_slli_si256(c, 2));
    c = _mm256_add_epi16(c, _mm256_slli_si256(c, 4));
    c = _mm256_add_epi16(c, _mm256_slli_si256(c, 8));
    __m128i loTot =
        _mm_broadcastw_epi16(_mm_srli_si128(_mm256_castsi256_si128(c), 14));
    c = _mm256_add_epi16(
        c, _mm256_inserti128_si256(_mm256_setzero_si256(), loTot, 1));
    c = _mm256_add_epi16(c, carry);
    _mm256_storeu_si256((__m256i *)(d + i), c);
    carry = _mm256_broadcastw_epi16(
        _mm_srli_si128(_mm256_extracti128_si256(c, 1), 14));
  }
#elif PLANAR_SSE2
  __m128i carry = _mm_set1_epi16((short)acc);
  for (; i + 8 <= n; i += 8) {
    __m128i c = _mm_sub_epi16(
        _mm_add_epi16(_mm_loadu_si128((const __m128i *)(res + i)),
                      _mm_loadu_si128((const __m128i *)(above + i))),
        _mm_loadu_si128((const __m128i *)(above + i - 1)));
    c = _mm_add_epi16(c, _mm_slli_si128(c, 2));
    c = _mm_add_epi16(c, _mm_slli_si128(c, 4));
    c = _mm_add_epi16(c, _mm_slli_si128(c, 8));
    c = _mm_add_epi16(c, carry);
    _mm_storeu_si128((__m128i *)(d + i), c);
    carry = _mm_set1_epi16((short)_mm_extract_epi16(c, 7));
  }
#elif PLANAR_NEON
  const uint16x8_t zero = vdupq_n_u16(0);
  uint16x8_t carry = vdupq_n_u16(acc);
  for (; i + 8 <= n; i += 8) {
    uint16x8_t c =
        vsubq_u16(vaddq_u16(vld1q_u16(res + i), vld1q_u16(above + i)),
                  vld1q_u16(above + i - 1));
    c = vaddq_u16(c, vextq_u16(zero, c, 7));
    c = vaddq_u16(c, vextq_u16(zero, c, 6));
    c = vaddq_u16(c, vextq_u16(zero, c, 4));
    c = vaddq_u16(c, carry);
    vst1q_u16(d + i, c);
    carry = vdupq_n_u16(vgetq_lane_u16(c, 7));
  }
#endif
  acc = d[i - 1];
  for (; i < n; ++i) {
    acc = (uint16_t)(acc + (uint16_t)(res[i] + above[i] - above[i - 1]));
    d[i] = acc;
  }
}

static void frow32(uint32_t *d, const uint32_t *res, const uint32_t *above,
                   size_t n) {
  uint32_t acc = res[0] + above[0];
  d[0] = acc;
  size_t i = 1;
#if PLANAR_AVX2
  __m256i carry = _mm256_set1_epi32((int)acc);
  for (; i + 8 <= n; i += 8) {
    __m256i c = _mm256_sub_epi32(
        _mm256_add_epi32(_mm256_loadu_si256((const __m256i *)(res + i)),
                         _mm256_loadu_si256((const __m256i *)(above + i))),
        _mm256_loadu_si256((const __m256i *)(above + i - 1)));
    c = _mm256_add_epi32(c, _mm256_slli_si256(c, 4));
    c = _mm256_add_epi32(c, _mm256_slli_si256(c, 8));
    __m128i loTot = _mm_shuffle_epi32(_mm256_castsi256_si128(c), 0xFF);
    c = _mm256_add_epi32(
        c, _mm256_inserti128_si256(_mm256_setzero_si256(), loTot, 1));
    c = _mm256_add_epi32(c, carry);
    _mm256_storeu_si256((__m256i *)(d + i), c);
    carry = _mm256_broadcastd_epi32(
        _mm_shuffle_epi32(_mm256_extracti128_si256(c, 1), 0xFF));
  }
#elif PLANAR_SSE2
  __m128i carry = _mm_set1_epi32((int)acc);
  for (; i + 4 <= n; i += 4) {
    __m128i c = _mm_sub_epi32(
        _mm_add_epi32(_mm_loadu_si128((const __m128i *)(res + i)),
                      _mm_loadu_si128((const __m128i *)(above + i))),
        _mm_loadu_si128((const __m128i *)(above + i - 1)));
    c = _mm_add_epi32(c, _mm_slli_si128(c, 4));
    c = _mm_add_epi32(c, _mm_slli_si128(c, 8));
    c = _mm_add_epi32(c, carry);
    _mm_storeu_si128((__m128i *)(d + i), c);
    carry = _mm_shuffle_epi32(c, _MM_SHUFFLE(3, 3, 3, 3));
  }
#elif PLANAR_NEON
  const uint32x4_t zero = vdupq_n_u32(0);
  uint32x4_t carry = vdupq_n_u32(acc);
  for (; i + 4 <= n; i += 4) {
    uint32x4_t c =
        vsubq_u32(vaddq_u32(vld1q_u32(res + i), vld1q_u32(above + i)),
                  vld1q_u32(above + i - 1));
    c = vaddq_u32(c, vextq_u32(zero, c, 3));
    c = vaddq_u32(c, vextq_u32(zero, c, 2));
    c = vaddq_u32(c, carry);
    vst1q_u32(d + i, c);
    carry = vdupq_n_u32(vgetq_lane_u32(c, 3));
  }
#endif
  acc = d[i - 1];
  for (; i < n; ++i) {
    acc += res[i] + above[i] - above[i - 1];
    d[i] = acc;
  }
}

static void frow64(uint64_t *d, const uint64_t *res, const uint64_t *above,
                   size_t n) {
  uint64_t acc = res[0] + above[0];
  d[0] = acc;
  for (size_t c = 1; c < n; ++c) {
    acc += res[c] + above[c] - above[c - 1];
    d[c] = acc;
  }
}

#define PLANAR_DEC(T, SCAN, FROW)                                              \
  do {                                                                         \
    T *d = (T *)dst;                                                           \
    const T *s = (const T *)src;                                               \
    SCAN(d, s, w);                                                             \
    for (size_t off = w; off < nbElts; off += w)                               \
      FROW(d + off, s + off, d + off - w, w);                                  \
  } while (0)

int planar_decode(void *dst, const void *src, size_t width, size_t nbElts,
                  size_t eltWidth) {
  // A width that does not divide nbElts would leave the last row short,
  // and the row loops below assume every row is complete.
  const size_t w = geozl_row_width(width, nbElts);
  if (w == 0)
    return 1;
  switch (eltWidth) {
  case 1:
    PLANAR_DEC(uint8_t, scan8, frow8);
    break;
  case 2:
    PLANAR_DEC(uint16_t, scan16, frow16);
    break;
  case 4:
    PLANAR_DEC(uint32_t, scan32, frow32);
    break;
  case 8:
    PLANAR_DEC(uint64_t, scan64, frow64);
    break;
  default:
    return 1; // eltWidth must be 1, 2, 4 or 8
  }
  return 0;
}

#undef PLANAR_DEC