#include "common/simd.h"

#include <stdlib.h>
#include <string.h>

/* What the build carries. AVX2 is compiled through a target attribute, so on
   x86 it is there whether or not the unit was built with -mavx2. */
#define GEOZL_BUILT_SSE2 GEOZL_SIMD_X86
#define GEOZL_BUILT_NEON GEOZL_SIMD_HAS_NEON
#define GEOZL_BUILT_AVX2 GEOZL_SIMD_CAN_AVX2

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

int geozl_simd_path = -1;

int geozl_simd_resolve(void) {
  const int ceiling = ceiling_from_env();
  unsigned allowed = 0;
  for (int p = GEOZL_SIMD_SCALAR; p <= ceiling; ++p)
    allowed |= GEOZL_SIMD_BIT(p);
  /* Benign race, every writer computes the same value. */
  geozl_simd_path = top_of(geozl_simd_built() & geozl_simd_cpu() & allowed);
  return geozl_simd_path;
}

int geozl_simd_active(void) { return geozl_simd_now(); }

const char *geozl_simd_name(int path) {
  switch (path) {
  case GEOZL_SIMD_SCALAR: return "scalar";
  case GEOZL_SIMD_SSE2:   return "sse2";
  case GEOZL_SIMD_AVX2:   return "avx2";
  case GEOZL_SIMD_NEON:   return "neon";
  default:                return "unknown";
  }
}
