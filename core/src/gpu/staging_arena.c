#include "staging_arena.h"

#include <stdint.h>
#include <stdio.h>

// Small allocations share a chunk until the first reset sizes the arena.
#define MIN_CHUNK ((size_t)1 << 20)

void geozl_gpu_arena_init(GeozlGpuArena *arena, GeozlGpuProgram *program) {
  *arena = (GeozlGpuArena){.program = program};
}

static void release(GeozlGpuArena *arena) {
  for (size_t i = 0; i < arena->count; ++i)
    geozl_gpu_staging_destroy(arena->chunks[i]);
  arena->count = 0;
}

void geozl_gpu_arena_destroy(GeozlGpuArena *arena) {
  if (arena != NULL)
    release(arena);
}

void geozl_gpu_arena_reset(GeozlGpuArena *arena) {
  arena->error[0] = '\0';
  if (arena->count < 2) {
    arena->used[0] = 0;
    return;
  }
  size_t total = 0;
  for (size_t i = 0; i < arena->count; ++i)
    total += arena->used[i];
  release(arena);
  // Headroom keeps a slightly larger batch from splitting the arena again.
  arena->reserve = total <= SIZE_MAX - total / 8 ? total + total / 8 : total;
}

void *geozl_gpu_arena_alloc(GeozlGpuArena *arena, size_t bytes) {
  const size_t mask = GEOZL_GPU_STAGING_ALIGNMENT - 1;
  if (bytes == 0 || bytes > SIZE_MAX - mask) {
    snprintf(arena->error, sizeof(arena->error), "invalid GPU staging size");
    return NULL;
  }
  const size_t size = (bytes + mask) & ~mask;
  if (arena->count != 0) {
    const size_t last = arena->count - 1;
    const size_t capacity = geozl_gpu_staging_capacity(arena->chunks[last]);
    if (size <= capacity - arena->used[last]) {
      void *memory = (uint8_t *)geozl_gpu_staging_data(arena->chunks[last]) +
                     arena->used[last];
      arena->used[last] += size;
      return memory;
    }
  }
  if (arena->count == GEOZL_GPU_ARENA_CHUNKS) {
    snprintf(arena->error, sizeof(arena->error),
             "GPU staging needs too many chunks");
    return NULL;
  }
  // A quarter more than a large request leaves room for the plans that follow
  // it, so a call fits one chunk and the next reset has nothing to merge.
  size_t capacity = size <= SIZE_MAX - size / 4 ? size + size / 4 : size;
  if (capacity < arena->reserve)
    capacity = arena->reserve;
  if (capacity < MIN_CHUNK)
    capacity = MIN_CHUNK;
  GeozlGpuStaging *chunk = geozl_gpu_staging_create(
      arena->program, capacity, arena->error, sizeof(arena->error));
  // A reserve the host cannot pin must not fail every later call.
  if (chunk == NULL && capacity > size && size >= MIN_CHUNK)
    chunk = geozl_gpu_staging_create(arena->program, size, arena->error,
                                     sizeof(arena->error));
  arena->reserve = 0;
  if (chunk == NULL)
    return NULL;
  arena->chunks[arena->count] = chunk;
  arena->used[arena->count++] = size;
  return geozl_gpu_staging_data(chunk);
}

GeozlGpuStaging *geozl_gpu_arena_staging(const GeozlGpuArena *arena,
                                         const void *memory) {
  for (size_t i = 0; i < arena->count; ++i) {
    const uintptr_t start = (uintptr_t)geozl_gpu_staging_data(arena->chunks[i]);
    if ((uintptr_t)memory >= start &&
        (uintptr_t)memory - start < arena->used[i])
      return arena->chunks[i];
  }
  return NULL;
}

static void *arena_host_alloc(void *arena, size_t bytes) {
  return geozl_gpu_arena_alloc(arena, bytes);
}

GeozlGpuHostAlloc geozl_gpu_arena_allocator(GeozlGpuArena *arena) {
  return (GeozlGpuHostAlloc){arena_host_alloc, arena};
}
