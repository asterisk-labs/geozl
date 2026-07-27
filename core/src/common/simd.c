#include "common/simd.h"

#include <stdlib.h>
#include <string.h>

/* Kept in step with scan.h by hand. scan.h is included inline everywhere and
   cannot pull this in without dragging it into every translation unit. */
#ifndef GEOZL_NO_SIMD
#if defined(__AVX2__)
#define GEOZL_BUILT_AVX2 1
#define GEOZL_BUILT_SSE2 1
#elif defined(__SSE2__) || defined(_M_X64) ||                                  \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#define GEOZL_BUILT_SSE2 1
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON)
#define GEOZL_BUILT_NEON 1
#endif
#endif

#ifndef GEOZL_BUILT_SSE2
#define GEOZL_BUILT_SSE2 0
#endif
#ifndef GEOZL_BUILT_AVX2
#define GEOZL_BUILT_AVX2 0
#endif
#ifndef GEOZL_BUILT_NEON
#define GEOZL_BUILT_NEON 0
#endif

unsigned geozl_simd_built(void) {
  unsigned m = GEOZL_SIMD_BIT(GEOZL_SIMD_SCALAR);
  if (GEOZL_BUILT_SSE2)
    m |= GEOZL_SIMD_BIT(GEOZL_SIMD_SSE2);
  if (GEOZL_BUILT_AVX2)
    m |= GEOZL_SIMD_BIT(GEOZL_SIMD_AVX2);
  if (GEOZL_BUILT_NEON)
    m |= GEOZL_SIMD_BIT(GEOZL_SIMD_NEON);
  return m;
}

static unsigned probe_cpu(void) {
  unsigned m = GEOZL_SIMD_BIT(GEOZL_SIMD_SCALAR);
#if defined(__aarch64__) || defined(_M_ARM64)
  m |= GEOZL_SIMD_BIT(GEOZL_SIMD_NEON); /* part of ARMv8 */
#elif defined(__x86_64__) || defined(_M_X64)
  m |= GEOZL_SIMD_BIT(GEOZL_SIMD_SSE2); /* part of the x86-64 baseline */
#if defined(__GNUC__) || defined(__clang__)
  if (__builtin_cpu_supports("avx2"))
    m |= GEOZL_SIMD_BIT(GEOZL_SIMD_AVX2);
#endif
#endif
  return m;
}

unsigned geozl_simd_cpu(void) {
  static unsigned cached; /* benign race, every writer computes the same */
  if (cached == 0)
    cached = probe_cpu();
  return cached;
}

static int top_of(unsigned mask) {
  int best = GEOZL_SIMD_SCALAR;
  for (int p = GEOZL_SIMD_SCALAR; p <= GEOZL_SIMD_NEON; ++p)
    if (mask & GEOZL_SIMD_BIT(p))
      best = p;
  return best;
}

static int ceiling_from_env(void) {
  const char *v = getenv("GEOZL_SIMD");
  if (v == NULL || v[0] == '\0')
    return GEOZL_SIMD_NEON;
  for (int p = GEOZL_SIMD_SCALAR; p <= GEOZL_SIMD_NEON; ++p)
    if (strcmp(v, geozl_simd_name(p)) == 0)
      return p;
  return GEOZL_SIMD_NEON;
}

int geozl_simd_active(void) {
  const int ceiling = ceiling_from_env();
  unsigned allowed = 0;
  for (int p = GEOZL_SIMD_SCALAR; p <= ceiling; ++p)
    allowed |= GEOZL_SIMD_BIT(p);
  return top_of(geozl_simd_built() & geozl_simd_cpu() & allowed);
}

const char *geozl_simd_name(int path) {
  switch (path) {
  case GEOZL_SIMD_SCALAR: return "scalar";
  case GEOZL_SIMD_SSE2:   return "sse2";
  case GEOZL_SIMD_AVX2:   return "avx2";
  case GEOZL_SIMD_NEON:   return "neon";
  default:                return "unknown";
  }
}
