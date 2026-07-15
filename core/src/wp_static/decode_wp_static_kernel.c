// Inverse wp_static predictor, W + K + residual with K = (cN*N + cNW*NW +
// cNE*NE + cNN*NN + round) >> shift over the row above. K has no left
// dependency, so each row folds K into the residual and a prefix sum resolves
// the W chain, the same scan as planar. K sums unsigned then shifts signed, so
// a u64 tile wraps at 2^64 rather than overflowing.
//
// For 1 and 2 byte samples the fold accumulates in 32 bits when shift <= 16.
// The output is <= 16 bits and only sum bits [shift, shift+16) reach it, so with
// shift <= 16 they stay below bit 32 and survive the wrap, matching the 64-bit
// fold exactly while letting the compiler vectorize the multiplies. A wider
// sample or a larger shift (no encoder emits one, a corrupt header might) takes
// the 64-bit path. @dst may alias @src.


#include "decode_wp_static_kernel.h"

#include <stddef.h>
#include <stdint.h>

#ifndef GEOZL_NO_SIMD // the ISA matrix sets this to force the scalar path
#if defined(__AVX2__)
#include <immintrin.h>
#define WP_STATIC_AVX2 1
#elif defined(__SSE2__) || defined(_M_X64) ||                                  \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <emmintrin.h>
#define WP_STATIC_SSE2 1
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON)
#include <arm_neon.h>
#define WP_STATIC_NEON 1
#endif
#endif // GEOZL_NO_SIMD

// Prefix sum, dst[i] = dst[i-1] + src[i], the W chain. src may alias dst.

static void scan8(uint8_t *dst, const uint8_t *src, size_t n) {
  size_t i = 0;
#if WP_STATIC_AVX2
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
#elif WP_STATIC_SSE2
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
#elif WP_STATIC_NEON
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
#endif
  uint8_t acc = (i > 0) ? dst[i - 1] : 0;
  for (; i < n; ++i) {
    acc = (uint8_t)(acc + src[i]);
    dst[i] = acc;
  }
}

static void scan16(uint16_t *dst, const uint16_t *src, size_t n) {
  size_t i = 0;
#if WP_STATIC_AVX2
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
#elif WP_STATIC_SSE2
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
#elif WP_STATIC_NEON
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
#endif
  uint16_t acc = (i > 0) ? dst[i - 1] : 0;
  for (; i < n; ++i) {
    acc = (uint16_t)(acc + src[i]);
    dst[i] = acc;
  }
}

static void scan32(uint32_t *dst, const uint32_t *src, size_t n) {
  size_t i = 0;
#if WP_STATIC_AVX2
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
#elif WP_STATIC_SSE2
  __m128i carry = _mm_setzero_si128();
  for (; i + 4 <= n; i += 4) {
    __m128i x = _mm_loadu_si128((const __m128i *)(src + i));
    x = _mm_add_epi32(x, _mm_slli_si128(x, 4));
    x = _mm_add_epi32(x, _mm_slli_si128(x, 8));
    x = _mm_add_epi32(x, carry);
    _mm_storeu_si128((__m128i *)(dst + i), x);
    carry = _mm_shuffle_epi32(x, _MM_SHUFFLE(3, 3, 3, 3));
  }
#elif WP_STATIC_NEON
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
#endif
  uint32_t acc = (i > 0) ? dst[i - 1] : 0;
  for (; i < n; ++i) {
    acc += src[i];
    dst[i] = acc;
  }
}

// Two elements per vector at 64 bits, not worth a vector cascade. Scalar.
static void scan64(uint64_t *dst, const uint64_t *src, size_t n) {
  uint64_t acc = 0;
  for (size_t i = 0; i < n; ++i) {
    acc += src[i];
    dst[i] = acc;
  }
}

// Fold d[c] = res[c] + K[c]. Edge taps (W, NW at column 0, NE at the last
// column, NN on row 1) are zero, so the two end columns are peeled. FAST picks
// the 32-bit interior for 1 and 2 byte samples; it compiles out otherwise.

#define WP_STATIC_KROW(T, NAME, FAST)                                          \
  static void NAME(T *d, const T *res, const T *ab, const T *ab2, size_t n,    \
                   uint64_t cN, uint64_t cNW, uint64_t cNE, uint64_t cNN,      \
                   uint64_t rnd, int sh) {                                     \
    {                                                                          \
      uint64_t acc = cN * ab[0] + (n > 1 ? cNE * ab[1] : 0) +                  \
                     (ab2 ? cNN * ab2[0] : 0) + rnd;                           \
      d[0] = (T)(res[0] + (T)((int64_t)acc >> sh));                            \
    }                                                                          \
    size_t c = 1;                                                              \
    if ((FAST) && sh <= 16) {                                                  \
      const uint32_t kN = (uint32_t)cN, kNW = (uint32_t)cNW,                   \
                     kNE = (uint32_t)cNE, kNN = (uint32_t)cNN,                 \
                     krd = (uint32_t)rnd;                                      \
      if (ab2) {                                                               \
        for (; c + 1 < n; ++c) {                                               \
          uint32_t acc = kN * ab[c] + kNW * ab[c - 1] + kNE * ab[c + 1] +      \
                         kNN * ab2[c] + krd;                                   \
          d[c] = (T)(res[c] + (T)((int32_t)acc >> sh));                        \
        }                                                                      \
      } else {                                                                 \
        for (; c + 1 < n; ++c) {                                               \
          uint32_t acc =                                                       \
              kN * ab[c] + kNW * ab[c - 1] + kNE * ab[c + 1] + krd;            \
          d[c] = (T)(res[c] + (T)((int32_t)acc >> sh));                        \
        }                                                                      \
      }                                                                        \
    } else if (ab2) {                                                          \
      for (; c + 1 < n; ++c) {                                                 \
        uint64_t acc = cN * ab[c] + cNW * ab[c - 1] + cNE * ab[c + 1] +        \
                       cNN * ab2[c] + rnd;                                     \
        d[c] = (T)(res[c] + (T)((int64_t)acc >> sh));                          \
      }                                                                        \
    } else {                                                                   \
      for (; c + 1 < n; ++c) {                                                 \
        uint64_t acc = cN * ab[c] + cNW * ab[c - 1] + cNE * ab[c + 1] + rnd;   \
        d[c] = (T)(res[c] + (T)((int64_t)acc >> sh));                          \
      }                                                                        \
    }                                                                          \
    if (n > 1) {                                                               \
      const size_t L = n - 1;                                                  \
      uint64_t acc =                                                           \
          cN * ab[L] + cNW * ab[L - 1] + (ab2 ? cNN * ab2[L] : 0) + rnd;       \
      d[L] = (T)(res[L] + (T)((int64_t)acc >> sh));                            \
    }                                                                          \
  }                                                                            \

WP_STATIC_KROW(uint8_t, krow8, 1)
WP_STATIC_KROW(uint16_t, krow16, 1)
WP_STATIC_KROW(uint32_t, krow32, 0)
WP_STATIC_KROW(uint64_t, krow64, 0)

#undef WP_STATIC_KROW

// Row 0 is a bare prefix sum; later rows fold K first. ab2 is null until row 2.

#define WP_STATIC_DEC(T, SCAN, KROW)                                           \
  do {                                                                         \
    T *d = (T *)dst;                                                           \
    const T *s = (const T *)src;                                               \
    const uint64_t cN = (uint64_t)coeffs[0], cNW = (uint64_t)coeffs[1],        \
                   cNE = (uint64_t)coeffs[2], cNN = (uint64_t)coeffs[3];       \
    const uint64_t rnd = shift ? (uint64_t)1 << (shift - 1) : 0;               \
    SCAN(d, s, w);                                                             \
    for (size_t off = w, r = 1; off < nbElts; off += w, ++r) {                 \
      const T *ab2 = (r >= 2) ? d + off - 2 * w : (const T *)0;                \
      KROW(d + off, s + off, d + off - w, ab2, w, cN, cNW, cNE, cNN, rnd,      \
           shift);                                                             \
      SCAN(d + off, d + off, w);                                               \
    }                                                                          \
  } while (0)

void wp_static_decode(void *dst, const void *src, size_t width, size_t nbElts,
                      size_t eltWidth, const int16_t coeffs[4], uint8_t shift) {
  if (nbElts == 0)
    return;
  const size_t w = (width == 0 || width > nbElts) ? nbElts : width;
  switch (eltWidth) {
  case 1:
    WP_STATIC_DEC(uint8_t, scan8, krow8);
    break;
  case 2:
    WP_STATIC_DEC(uint16_t, scan16, krow16);
    break;
  case 4:
    WP_STATIC_DEC(uint32_t, scan32, krow32);
    break;
  case 8:
    WP_STATIC_DEC(uint64_t, scan64, krow64);
    break;
  default:
    break;
  }
}

#undef WP_STATIC_DEC