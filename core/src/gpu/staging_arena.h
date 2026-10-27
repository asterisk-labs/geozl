#ifndef GEOZL_GPU_STAGING_ARENA_H
#define GEOZL_GPU_STAGING_ARENA_H

#include "host_alloc.h"
#include "runtime.h"

#include <stddef.h>

#define GEOZL_GPU_ARENA_CHUNKS 16

// Bump allocator over staging that a decoder keeps between calls. reset merges
// the chunks the last call needed into one, so batches of a similar size stop
// allocating staging after the first call. The arena keeps its largest batch's
// staging until it is destroyed.
typedef struct {
  GeozlGpuProgram *program;
  GeozlGpuStaging *chunks[GEOZL_GPU_ARENA_CHUNKS];
  size_t used[GEOZL_GPU_ARENA_CHUNKS];
  size_t count;
  size_t reserve;
  char error[256];
} GeozlGpuArena;

void geozl_gpu_arena_init(GeozlGpuArena *arena, GeozlGpuProgram *program);
void geozl_gpu_arena_destroy(GeozlGpuArena *arena);
void geozl_gpu_arena_reset(GeozlGpuArena *arena);

// Returns memory aligned to GEOZL_GPU_STAGING_ALIGNMENT, or NULL with
// arena->error set.
void *geozl_gpu_arena_alloc(GeozlGpuArena *arena, size_t bytes);
GeozlGpuStaging *geozl_gpu_arena_staging(const GeozlGpuArena *arena,
                                         const void *memory);
GeozlGpuHostAlloc geozl_gpu_arena_allocator(GeozlGpuArena *arena);

// A kernel input of count elements read in place from arena memory. Kernels
// need a bound buffer even when they skip it, so it spans at least one element.
static inline GeozlGpuBuffer geozl_gpu_arena_input(const GeozlGpuArena *arena,
                                                   const void *host,
                                                   size_t count,
                                                   size_t elementBytes) {
  return (GeozlGpuBuffer){.host = (void *)host,
                          .bytes = (count == 0 ? 1 : count) * elementBytes,
                          .elementBytes = elementBytes,
                          .flags = GEOZL_GPU_COPY_IN,
                          .alias = -1,
                          .staging = geozl_gpu_arena_staging(arena, host)};
}

static inline const char *geozl_gpu_arena_error(const GeozlGpuArena *arena) {
  return arena->error[0] != '\0' ? arena->error : "out of memory";
}

#endif // GEOZL_GPU_STAGING_ARENA_H
