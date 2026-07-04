// Inverse wp_static predictor. The kernel K over the row above has no left
// dependency, so it is a plain map that vectorizes, and the remaining W chain is
// a prefix sum, the same scan as planar. K is accumulated in int32 for 8 and
// 16-bit samples, int64 for 32 and 64-bit. @dst may alias @src.

#include "decode_wp_static_kernel.h"

#include <stddef.h>

#if defined(__AVX2__)
#    include <immintrin.h>
#    define WP_STATIC_AVX2 1
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#    include <emmintrin.h>
#    define WP_STATIC_SSE2 1
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON)
#    include <arm_neon.h>
#    define WP_STATIC_NEON 1
#endif

// Prefix sum, dst[i] = dst[i-1] + src[i], the W chain. src may alias dst.

static void scan8(uint8_t* dst, const uint8_t* src, size_t n)
{
    size_t i = 0;
#if WP_STATIC_AVX2
    __m256i carry = _mm256_setzero_si256();
    for (; i + 32 <= n; i += 32) {
        __m256i x = _mm256_loadu_si256((const __m256i*)(src + i));
        x = _mm256_add_epi8(x, _mm256_slli_si256(x, 1));
        x = _mm256_add_epi8(x, _mm256_slli_si256(x, 2));
        x = _mm256_add_epi8(x, _mm256_slli_si256(x, 4));
        x = _mm256_add_epi8(x, _mm256_slli_si256(x, 8));
        __m128i loTot = _mm_broadcastb_epi8(_mm_srli_si128(_mm256_castsi256_si128(x), 15));
        x = _mm256_add_epi8(x, _mm256_inserti128_si256(_mm256_setzero_si256(), loTot, 1));
        x = _mm256_add_epi8(x, carry);
        _mm256_storeu_si256((__m256i*)(dst + i), x);
        carry = _mm256_broadcastb_epi8(_mm_srli_si128(_mm256_extracti128_si256(x, 1), 15));
    }
#elif WP_STATIC_SSE2
    __m128i carry = _mm_setzero_si128();
    for (; i + 16 <= n; i += 16) {
        __m128i x = _mm_loadu_si128((const __m128i*)(src + i));
        x = _mm_add_epi8(x, _mm_slli_si128(x, 1));
        x = _mm_add_epi8(x, _mm_slli_si128(x, 2));
        x = _mm_add_epi8(x, _mm_slli_si128(x, 4));
        x = _mm_add_epi8(x, _mm_slli_si128(x, 8));
        x = _mm_add_epi8(x, carry);
        _mm_storeu_si128((__m128i*)(dst + i), x);
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
    for (; i < n; ++i) { acc = (uint8_t)(acc + src[i]); dst[i] = acc; }
}

static void scan16(uint16_t* dst, const uint16_t* src, size_t n)
{
    size_t i = 0;
#if WP_STATIC_AVX2
    __m256i carry = _mm256_setzero_si256();
    for (; i + 16 <= n; i += 16) {
        __m256i x = _mm256_loadu_si256((const __m256i*)(src + i));
        x = _mm256_add_epi16(x, _mm256_slli_si256(x, 2));
        x = _mm256_add_epi16(x, _mm256_slli_si256(x, 4));
        x = _mm256_add_epi16(x, _mm256_slli_si256(x, 8));
        __m128i loTot = _mm_broadcastw_epi16(_mm_srli_si128(_mm256_castsi256_si128(x), 14));
        x = _mm256_add_epi16(x, _mm256_inserti128_si256(_mm256_setzero_si256(), loTot, 1));
        x = _mm256_add_epi16(x, carry);
        _mm256_storeu_si256((__m256i*)(dst + i), x);
        carry = _mm256_broadcastw_epi16(_mm_srli_si128(_mm256_extracti128_si256(x, 1), 14));
    }
#elif WP_STATIC_SSE2
    __m128i carry = _mm_setzero_si128();
    for (; i + 8 <= n; i += 8) {
        __m128i x = _mm_loadu_si128((const __m128i*)(src + i));
        x = _mm_add_epi16(x, _mm_slli_si128(x, 2));
        x = _mm_add_epi16(x, _mm_slli_si128(x, 4));
        x = _mm_add_epi16(x, _mm_slli_si128(x, 8));
        x = _mm_add_epi16(x, carry);
        _mm_storeu_si128((__m128i*)(dst + i), x);
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
    for (; i < n; ++i) { acc = (uint16_t)(acc + src[i]); dst[i] = acc; }
}

static void scan32(uint32_t* dst, const uint32_t* src, size_t n)
{
    size_t i = 0;
#if WP_STATIC_AVX2
    __m256i carry = _mm256_setzero_si256();
    for (; i + 8 <= n; i += 8) {
        __m256i x = _mm256_loadu_si256((const __m256i*)(src + i));
        x = _mm256_add_epi32(x, _mm256_slli_si256(x, 4));
        x = _mm256_add_epi32(x, _mm256_slli_si256(x, 8));
        __m128i loTot = _mm_shuffle_epi32(_mm256_castsi256_si128(x), 0xFF);
        x = _mm256_add_epi32(x, _mm256_inserti128_si256(_mm256_setzero_si256(), loTot, 1));
        x = _mm256_add_epi32(x, carry);
        _mm256_storeu_si256((__m256i*)(dst + i), x);
        carry = _mm256_broadcastd_epi32(_mm_shuffle_epi32(_mm256_extracti128_si256(x, 1), 0xFF));
    }
#elif WP_STATIC_SSE2
    __m128i carry = _mm_setzero_si128();
    for (; i + 4 <= n; i += 4) {
        __m128i x = _mm_loadu_si128((const __m128i*)(src + i));
        x = _mm_add_epi32(x, _mm_slli_si128(x, 4));
        x = _mm_add_epi32(x, _mm_slli_si128(x, 8));
        x = _mm_add_epi32(x, carry);
        _mm_storeu_si128((__m128i*)(dst + i), x);
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
    for (; i < n; ++i) { acc += src[i]; dst[i] = acc; }
}

// u64 vectors hold too few elements to pay for a vector prefix sum.
static void scan64(uint64_t* dst, const uint64_t* src, size_t n)
{
    uint64_t acc = 0;
    for (size_t i = 0; i < n; ++i) { acc += src[i]; dst[i] = acc; }
}

// Fold the kernel into the residual, c[i] = res[i] + K[i]. K reads only the row
// above, so the interior span has no dependency and vectorizes. Edge taps are
// zero, and ab2, the row two above, is null on row one.

#define WP_STATIC_KROW(T, ACC, NAME)                                          \
    static void NAME(T* d, const T* res, const T* ab, const T* ab2, size_t n, \
                     ACC cN, ACC cNW, ACC cNE, ACC cNN, ACC rnd, int sh)       \
    {                                                                          \
        {                                                                      \
            ACC K = (cN * (ACC)ab[0] + (n > 1 ? cNE * (ACC)ab[1] : 0)          \
                     + (ab2 ? cNN * (ACC)ab2[0] : 0) + rnd) >> sh;             \
            d[0] = (T)(res[0] + (T)K);                                         \
        }                                                                      \
        size_t c = 1;                                                          \
        if (ab2) {                                                             \
            for (; c + 1 < n; ++c) {                                           \
                ACC K = (cN * (ACC)ab[c] + cNW * (ACC)ab[c - 1]                \
                         + cNE * (ACC)ab[c + 1] + cNN * (ACC)ab2[c] + rnd)     \
                      >> sh;                                                   \
                d[c] = (T)(res[c] + (T)K);                                     \
            }                                                                  \
        } else {                                                               \
            for (; c + 1 < n; ++c) {                                           \
                ACC K = (cN * (ACC)ab[c] + cNW * (ACC)ab[c - 1]                \
                         + cNE * (ACC)ab[c + 1] + rnd) >> sh;                  \
                d[c] = (T)(res[c] + (T)K);                                     \
            }                                                                  \
        }                                                                      \
        if (n > 1) {                                                           \
            const size_t L = n - 1;                                            \
            ACC K = (cN * (ACC)ab[L] + cNW * (ACC)ab[L - 1]                    \
                     + (ab2 ? cNN * (ACC)ab2[L] : 0) + rnd) >> sh;             \
            d[L] = (T)(res[L] + (T)K);                                         \
        }                                                                      \
    }

WP_STATIC_KROW(uint8_t,  int32_t, krow8)
WP_STATIC_KROW(uint16_t, int32_t, krow16)
WP_STATIC_KROW(uint32_t, int64_t, krow32)
WP_STATIC_KROW(uint64_t, int64_t, krow64)

#undef WP_STATIC_KROW

#define WP_STATIC_DEC(T, ACC, SCAN, KROW)                                     \
    do {                                                                       \
        T* d       = (T*)dst;                                                  \
        const T* s = (const T*)src;                                            \
        const ACC cN = coeffs[0], cNW = coeffs[1], cNE = coeffs[2],            \
                  cNN = coeffs[3];                                             \
        const ACC rnd = shift ? (ACC)1 << (shift - 1) : 0;                     \
        SCAN(d, s, w);                                                         \
        for (size_t off = w, r = 1; off < nbElts; off += w, ++r) {            \
            const T* ab2 = (r >= 2) ? d + off - 2 * w : (const T*)0;          \
            KROW(d + off, s + off, d + off - w, ab2, w, cN, cNW, cNE, cNN,     \
                 rnd, shift);                                                  \
            SCAN(d + off, d + off, w);                                         \
        }                                                                     \
    } while (0)

void wp_static_decode(
        void* dst,
        const void* src,
        size_t width,
        size_t nbElts,
        size_t eltWidth,
        const int16_t coeffs[4],
        uint8_t shift)
{
    if (nbElts == 0)
        return;
    const size_t w = (width == 0 || width > nbElts) ? nbElts : width;
    switch (eltWidth) {
        case 1: WP_STATIC_DEC(uint8_t,  int32_t, scan8,  krow8);  break;
        case 2: WP_STATIC_DEC(uint16_t, int32_t, scan16, krow16); break;
        case 4: WP_STATIC_DEC(uint32_t, int64_t, scan32, krow32); break;
        case 8: WP_STATIC_DEC(uint64_t, int64_t, scan64, krow64); break;
        default: break;
    }
}

#undef WP_STATIC_DEC
