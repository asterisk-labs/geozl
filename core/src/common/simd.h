/* Which vector paths a build carries, which the machine can run, and which one
   actually runs. The three differ, and only the gap between them shows that a
   build lost a fast path, since the frames come out the same either way.

   The values cross into Python, so a published number keeps its meaning. */

#ifndef GEOZL_COMMON_SIMD_H
#define GEOZL_COMMON_SIMD_H

#define GEOZL_SIMD_SCALAR 0
#define GEOZL_SIMD_SSE2 1
#define GEOZL_SIMD_AVX2 2
#define GEOZL_SIMD_NEON 3

#define GEOZL_SIMD_BIT(p) (1u << (p))

unsigned geozl_simd_built(void);
unsigned geozl_simd_cpu(void);

/* Best path that is both built and available, lowered by GEOZL_SIMD when that
   names a narrower one. A value it does not recognise is ignored. */
int geozl_simd_active(void);

const char *geozl_simd_name(int path);

#endif /* GEOZL_COMMON_SIMD_H */
