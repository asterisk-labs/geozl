#include "pivco_plan.h"

#include <limits.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  const CapturedFrame *frames;
  const GeozlGpuFrameShape *shapes;
  size_t first;
  size_t count;
  const uint8_t *payload;
  size_t payloadSize;
  size_t elementBytes;
  PivcoGpuPlan plan;
  int failed;
  int threaded;
} PlanWorker;

static void trim_lane_padding(PivcoGpuPlan *plan) {
  while (plan->laneBlockCount != 0 &&
         plan->laneBlocks[plan->laneBlockCount - 1].outputBase ==
             plan->laneBytes)
    --plan->laneBlockCount;
}

static void *build_chunk(void *opaque) {
  PlanWorker *worker = opaque;
  worker->failed = pivco_gpu_plan_build(
      &worker->plan, worker->frames + worker->first,
      worker->shapes + worker->first, worker->count, worker->payload,
      worker->payloadSize, worker->elementBytes);
  if (!worker->failed)
    trim_lane_padding(&worker->plan);
  return NULL;
}

static void free_workers(PlanWorker *workers, size_t count) {
  if (workers == NULL)
    return;
  for (size_t i = 0; i < count; ++i)
    pivco_gpu_plan_free(&workers[i].plan);
  free(workers);
}

static int add_size(size_t *total, size_t value) {
  if (value > SIZE_MAX - *total)
    return 1;
  *total += value;
  return 0;
}

static int allocate_array(void **array, size_t count, size_t elementBytes,
                          const GeozlGpuHostAlloc *alloc) {
  *array = geozl_gpu_host_calloc(alloc, count == 0 ? 1 : count, elementBytes);
  return *array == NULL;
}

#define ALLOC(plan, member, count, alloc)                                      \
  allocate_array((void **)&(plan)->member, (count), sizeof *(plan)->member,    \
                 (alloc))

static int allocate_plan(PivcoGpuPlan *plan, const GeozlGpuHostAlloc *alloc) {
  return ALLOC(plan, symbols, plan->symbolCount < 4 ? 4 : plan->symbolCount,
               alloc) ||
         ALLOC(plan, trees, plan->treeCount, alloc) ||
         ALLOC(plan, nodes, plan->nodeCount, alloc) ||
         ALLOC(plan, pivcoBlocks, plan->pivcoBlockCount, alloc) ||
         ALLOC(plan, laneBlocks, plan->laneBlockCount + 4, alloc) ||
         ALLOC(plan, joins, plan->joinCount, alloc);
}

static int add_u32(uint32_t *value, size_t offset) {
  if (offset > UINT32_MAX - *value)
    return 1;
  *value += (uint32_t)offset;
  return 0;
}

static int merge_plans(PivcoGpuPlan *plan, const PlanWorker workers[],
                       size_t workerCount, const GeozlGpuHostAlloc *alloc) {
  *plan = (PivcoGpuPlan){.borrowed = alloc != NULL};
  for (size_t i = 0; i < workerCount; ++i) {
    const PivcoGpuPlan *src = &workers[i].plan;
    if (add_size(&plan->symbolCount, src->symbolCount) ||
        add_size(&plan->treeCount, src->treeCount) ||
        add_size(&plan->nodeCount, src->nodeCount) ||
        add_size(&plan->bitmapCount, src->bitmapCount) ||
        add_size(&plan->pivcoBlockCount, src->pivcoBlockCount) ||
        add_size(&plan->laneBlockCount, src->laneBlockCount) ||
        add_size(&plan->joinCount, src->joinCount) ||
        add_size(&plan->laneBytes, src->laneBytes) ||
        add_size(&plan->pivcoLaneCount, src->pivcoLaneCount) ||
        add_size(&plan->pforLaneCount, src->pforLaneCount))
      goto overflow;
  }
  if (plan->symbolCount > UINT32_MAX || plan->treeCount > UINT32_MAX ||
      plan->nodeCount > UINT32_MAX ||
      plan->laneBytes > (size_t)UINT32_MAX - 256 ||
      plan->laneBlockCount > SIZE_MAX - 4)
    goto overflow;
  if (allocate_plan(plan, alloc)) {
    pivco_gpu_plan_free(plan);
    plan->error = "cannot allocate the PivCo plan";
    plan->errorFrame = SIZE_MAX;
    plan->errorLane = SIZE_MAX;
    return 1;
  }

  size_t symbolBase = 0, treeBase = 0, nodeBase = 0;
  size_t pivcoBase = 0, laneBlockBase = 0, joinBase = 0, laneBase = 0;
  for (size_t i = 0; i < workerCount; ++i) {
    const PivcoGpuPlan *src = &workers[i].plan;
    if (src->symbolCount != 0)
      memcpy(plan->symbols + symbolBase, src->symbols, src->symbolCount);
    if (src->treeCount != 0)
      memcpy(plan->trees + treeBase, src->trees,
             src->treeCount * sizeof(*src->trees));
    if (src->nodeCount != 0)
      memcpy(plan->nodes + nodeBase, src->nodes,
             src->nodeCount * sizeof(*src->nodes));

    for (size_t j = 0; j < src->nodeCount; ++j) {
      if (add_u32(&plan->nodes[nodeBase + j].output, laneBase))
        goto overflow;
    }
    for (size_t j = 0; j < src->pivcoBlockCount; ++j) {
      GpuPivcoBlock block = src->pivcoBlocks[j];
      if (add_u32(&block.nodeBase, nodeBase) ||
          add_u32(&block.treeBase, treeBase) ||
          add_u32(&block.symbolBase, symbolBase) ||
          add_u32(&block.outBase, laneBase))
        goto overflow;
      plan->pivcoBlocks[pivcoBase + j] = block;
    }
    for (size_t j = 0; j < src->laneBlockCount; ++j) {
      GpuLaneBlock block = src->laneBlocks[j];
      if (add_u32(&block.outputBase, laneBase))
        goto overflow;
      plan->laneBlocks[laneBlockBase + j] = block;
    }
    for (size_t j = 0; j < src->joinCount; ++j) {
      GpuJoinTask join = src->joins[j];
      if (add_u32(&join.laneBase, laneBase))
        goto overflow;
      plan->joins[joinBase + j] = join;
    }

    symbolBase += src->symbolCount;
    treeBase += src->treeCount;
    nodeBase += src->nodeCount;
    pivcoBase += src->pivcoBlockCount;
    laneBlockBase += src->laneBlockCount;
    joinBase += src->joinCount;
    laneBase += src->laneBytes;
  }

  while (plan->laneBlockCount % 4 && plan->laneBlockCount != 0) {
    plan->laneBlocks[plan->laneBlockCount++] = (GpuLaneBlock){
        plan->laneBlocks[0].payloadOffset, (uint32_t)plan->laneBytes};
  }
  return 0;

overflow:
  pivco_gpu_plan_free(plan);
  plan->error = "parallel GPU plan exceeds its limits";
  plan->errorFrame = SIZE_MAX;
  plan->errorLane = SIZE_MAX;
  return 1;
}

int pivco_gpu_plan_build_parallel(PivcoGpuPlan *plan,
                                  const CapturedFrame *frames,
                                  const GeozlGpuFrameShape *shapes,
                                  size_t frameCount, const uint8_t *payload,
                                  size_t payloadSize, size_t elementBytes,
                                  size_t threadCount,
                                  const GeozlGpuHostAlloc *alloc) {
  if (plan == NULL || frames == NULL || shapes == NULL || frameCount == 0)
    return pivco_gpu_plan_build(plan, frames, shapes, frameCount, payload,
                                payloadSize, elementBytes);
  if (threadCount > frameCount)
    threadCount = frameCount;
  if (threadCount == 0)
    threadCount = 1;

  PlanWorker *workers = calloc(threadCount, sizeof(*workers));
  pthread_t *threads = calloc(threadCount, sizeof(*threads));
  if (workers == NULL || threads == NULL) {
    free(workers);
    free(threads);
    *plan = (PivcoGpuPlan){.error = "out of memory",
                           .errorFrame = SIZE_MAX,
                           .errorLane = SIZE_MAX};
    return 1;
  }

  // A worker that cannot get a thread runs on this one.
  size_t first = 0;
  for (size_t i = 0; i < threadCount; ++i) {
    const size_t count =
        (frameCount - first + threadCount - i - 1) / (threadCount - i);
    workers[i] = (PlanWorker){frames,      shapes,       first, count, payload,
                              payloadSize, elementBytes, {0},   0,     0};
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
      const size_t errorLane = workers[i].plan.errorLane;
      if (errorFrame != SIZE_MAX)
        errorFrame += workers[i].first;
      free_workers(workers, threadCount);
      *plan = (PivcoGpuPlan){
          .error = message, .errorFrame = errorFrame, .errorLane = errorLane};
      return 1;
    }
  }

  const int failed = merge_plans(plan, workers, threadCount, alloc);
  free_workers(workers, threadCount);
  return failed;
}

#undef ALLOC
