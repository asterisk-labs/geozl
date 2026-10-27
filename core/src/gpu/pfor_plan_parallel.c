#include "pfor_plan.h"

#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  const PforGpuFrame *frames;
  size_t first;
  size_t count;
  const uint8_t *payload;
  size_t payloadSize;
  size_t elementBytes;
  PforGpuPlan plan;
  int failed;
  int threaded;
} PlanWorker;

static void *build_chunk(void *opaque) {
  PlanWorker *worker = opaque;
  worker->failed = pfor_gpu_plan_build(
      &worker->plan, worker->frames + worker->first, worker->count,
      worker->payload, worker->payloadSize, worker->elementBytes);
  return NULL;
}

static void free_workers(PlanWorker *workers, size_t count) {
  if (workers == NULL)
    return;
  for (size_t i = 0; i < count; ++i)
    pfor_gpu_plan_free(&workers[i].plan);
  free(workers);
}

static int add_size(size_t *total, size_t value) {
  if (value > SIZE_MAX - *total)
    return 1;
  *total += value;
  return 0;
}

static int merge_plans(PforGpuPlan *plan, const PlanWorker workers[],
                       size_t workerCount, const GeozlGpuHostAlloc *alloc) {
  *plan = (PforGpuPlan){.borrowed = alloc != NULL};
  for (size_t i = 0; i < workerCount; ++i) {
    const PforGpuPlan *src = &workers[i].plan;
    if (add_size(&plan->realBlockCount, src->realBlockCount) ||
        add_size(&plan->blocksWithExceptions, src->blocksWithExceptions) ||
        add_size(&plan->exceptionCount, src->exceptionCount))
      goto overflow;
  }
  if (plan->realBlockCount > SIZE_MAX - 3)
    goto overflow;
  plan->blockCount = (plan->realBlockCount + 3) / 4 * 4;
  plan->blocks =
      geozl_gpu_host_calloc(alloc, plan->blockCount, sizeof(*plan->blocks));
  if (plan->blocks == NULL) {
    pfor_gpu_plan_free(plan);
    plan->error = "cannot allocate the PFOR plan";
    plan->errorFrame = SIZE_MAX;
    plan->errorBlock = SIZE_MAX;
    return 1;
  }

  size_t at = 0;
  for (size_t i = 0; i < workerCount; ++i) {
    const PforGpuPlan *src = &workers[i].plan;
    memcpy(plan->blocks + at, src->blocks,
           src->realBlockCount * sizeof(*src->blocks));
    at += src->realBlockCount;
  }
  return 0;

overflow:
  pfor_gpu_plan_free(plan);
  plan->error = "parallel PFOR plan exceeds its limits";
  plan->errorFrame = SIZE_MAX;
  plan->errorBlock = SIZE_MAX;
  return 1;
}

int pfor_gpu_plan_build_parallel(PforGpuPlan *plan, const PforGpuFrame *frames,
                                 size_t frameCount, const uint8_t *payload,
                                 size_t payloadSize, size_t elementBytes,
                                 size_t threadCount,
                                 const GeozlGpuHostAlloc *alloc) {
  if (plan == NULL || frames == NULL || frameCount == 0)
    return pfor_gpu_plan_build(plan, frames, frameCount, payload, payloadSize,
                               elementBytes);
  if (threadCount > frameCount)
    threadCount = frameCount;
  if (threadCount == 0)
    threadCount = 1;

  PlanWorker *workers = calloc(threadCount, sizeof(*workers));
  pthread_t *threads = calloc(threadCount, sizeof(*threads));
  if (workers == NULL || threads == NULL) {
    free(workers);
    free(threads);
    *plan = (PforGpuPlan){.error = "out of memory",
                          .errorFrame = SIZE_MAX,
                          .errorBlock = SIZE_MAX};
    return 1;
  }

  // A worker that cannot get a thread runs on this one.
  size_t first = 0;
  for (size_t i = 0; i < threadCount; ++i) {
    const size_t count =
        (frameCount - first + threadCount - i - 1) / (threadCount - i);
    workers[i] = (PlanWorker){frames,       first, count, payload, payloadSize,
                              elementBytes, {0},   0,     0};
    first += count;
    workers[i].threaded =
        threadCount > 1 &&
        pthread_create(&threads[i], NULL, build_chunk, &workers[i]) == 0;
    if (!workers[i].threaded)
      build_chunk(&workers[i]);
  }
  for (size_t i = 0; i < threadCount; ++i) {
    if (workers[i].threaded)
      pthread_join(threads[i], NULL);
  }
  free(threads);

  for (size_t i = 0; i < threadCount; ++i) {
    if (workers[i].failed) {
      const char *message = workers[i].plan.error;
      size_t errorFrame = workers[i].plan.errorFrame;
      const size_t errorBlock = workers[i].plan.errorBlock;
      if (errorFrame != SIZE_MAX)
        errorFrame += workers[i].first;
      free_workers(workers, threadCount);
      *plan = (PforGpuPlan){
          .error = message, .errorFrame = errorFrame, .errorBlock = errorBlock};
      return 1;
    }
  }

  const int failed = merge_plans(plan, workers, threadCount, alloc);
  free_workers(workers, threadCount);
  return failed;
}
