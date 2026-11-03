#ifndef GEOZL_GPU_PLANAR_SCAN_H
#define GEOZL_GPU_PLANAR_SCAN_H

#include "host_alloc.h"

#include <stddef.h>
#include <stdint.h>

// Planes wider than one segment of the scan kernels take the row and column
// passes instead of scan_planes. CUDA caps grid Y at 65535, and those passes
// cover four rows or 128 columns per group.
#define GEOZL_GPU_SCAN_SEGMENT 256
#define GEOZL_GPU_SCAN_MAX_ROWS ((size_t)65535 * 4)
#define GEOZL_GPU_SCAN_MAX_WIDTH ((size_t)65535 * 128)

typedef struct {
  size_t outOff;
  size_t nbElts;
  size_t width;
  uint32_t planes;
} GeozlGpuFrameShape;

// Mirrors plane_task in gpu/kernels/programs/pfor.slang and pivco.slang.
typedef struct {
  uint32_t outBase;
  uint32_t width;
  uint32_t rows;
} GeozlGpuPlaneTask;

typedef struct {
  GeozlGpuPlaneTask *tasks;
  size_t count;
  uint32_t maxWidth;
  uint32_t maxRows; // over planes wider than GEOZL_GPU_SCAN_SEGMENT
  int borrowed;     // tasks belong to the allocator the plan was built with
} GeozlGpuPlanePlan;

// Zero when a planar frame with checked geometry belongs on the host. Planes
// shorter than a segment could plan more task bytes than the frame decodes to.
// With gridLimited, as on CUDA, wide planes past the limits above do not fit a
// grid.
int geozl_gpu_plane_shape_fits(const GeozlGpuFrameShape *frame,
                               int gridLimited);

// Puts the tasks in memory from alloc; NULL means calloc.
int geozl_gpu_plane_plan_build(GeozlGpuPlanePlan *plan,
                               const GeozlGpuFrameShape *frames,
                               size_t frameCount,
                               const GeozlGpuHostAlloc *alloc);
void geozl_gpu_plane_plan_free(GeozlGpuPlanePlan *plan);

#endif // GEOZL_GPU_PLANAR_SCAN_H
