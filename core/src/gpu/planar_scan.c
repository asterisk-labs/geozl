#include "planar_scan.h"

#include <limits.h>
#include <stdlib.h>

_Static_assert(sizeof(GeozlGpuPlaneTask) == 12,
               "GeozlGpuPlaneTask must match Slang");

void geozl_gpu_plane_plan_free(GeozlGpuPlanePlan *plan) {
  if (plan == NULL)
    return;
  if (!plan->borrowed)
    free(plan->tasks);
  *plan = (GeozlGpuPlanePlan){0};
}

int geozl_gpu_plane_plan_build(GeozlGpuPlanePlan *plan,
                               const GeozlGpuFrameShape *frames,
                               size_t frameCount,
                               const GeozlGpuHostAlloc *alloc) {
  if (plan == NULL)
    return 1;
  *plan = (GeozlGpuPlanePlan){0};
  if (frames == NULL || frameCount == 0)
    return 1;

  size_t count = 0;
  for (size_t i = 0; i < frameCount; ++i) {
    const GeozlGpuFrameShape *frame = &frames[i];
    if (frame->nbElts == 0 || frame->planes == 0 || frame->width == 0 ||
        frame->nbElts % frame->planes != 0 ||
        frame->nbElts / frame->planes % frame->width != 0 ||
        frame->planes > SIZE_MAX - count)
      return 1;
    count += frame->planes;
  }
  if (count > SIZE_MAX - 3)
    return 1;
  const size_t padded = (count + 3) / 4 * 4;
  plan->tasks = geozl_gpu_host_calloc(alloc, padded, sizeof(*plan->tasks));
  if (plan->tasks == NULL)
    return 1;
  plan->borrowed = alloc != NULL;

  for (size_t i = 0; i < frameCount; ++i) {
    const GeozlGpuFrameShape *frame = &frames[i];
    const size_t planeElts = frame->nbElts / frame->planes;
    const size_t rows = planeElts / frame->width;
    for (uint32_t plane = 0; plane < frame->planes; ++plane) {
      if (plane > SIZE_MAX / planeElts ||
          frame->outOff > SIZE_MAX - (size_t)plane * planeElts) {
        geozl_gpu_plane_plan_free(plan);
        return 1;
      }
      const size_t outBase = frame->outOff + (size_t)plane * planeElts;
      if (outBase > UINT32_MAX || frame->width > UINT32_MAX ||
          rows > UINT32_MAX) {
        geozl_gpu_plane_plan_free(plan);
        return 1;
      }
      GeozlGpuPlaneTask *task = &plan->tasks[plan->count++];
      *task = (GeozlGpuPlaneTask){(uint32_t)outBase, (uint32_t)frame->width,
                                  (uint32_t)rows};
      if (task->width > plan->maxWidth)
        plan->maxWidth = task->width;
      if (task->width > GEOZL_GPU_SCAN_SEGMENT && task->rows > plan->maxRows)
        plan->maxRows = task->rows;
    }
  }
  plan->count = padded;
  return 0;
}
