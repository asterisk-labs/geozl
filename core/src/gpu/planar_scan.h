#ifndef GEOZL_GPU_PLANAR_SCAN_H
#define GEOZL_GPU_PLANAR_SCAN_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
  size_t outOff;
  size_t nbElts;
  size_t width;
  uint32_t planes;
} GeozlGpuFrameShape;

// Mirrors plane_task in planar_scan.slang.
typedef struct {
  uint32_t outBase;
  uint32_t width;
  uint32_t rows;
} GeozlGpuPlaneTask;

typedef struct {
  GeozlGpuPlaneTask *tasks;
  size_t count;
  uint32_t maxWidth;
  uint32_t maxRows;
} GeozlGpuPlanePlan;

int geozl_gpu_plane_plan_build(GeozlGpuPlanePlan *plan,
                               const GeozlGpuFrameShape *frames,
                               size_t frameCount);
void geozl_gpu_plane_plan_free(GeozlGpuPlanePlan *plan);

#endif // GEOZL_GPU_PLANAR_SCAN_H
