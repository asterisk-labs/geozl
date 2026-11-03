#include "gpu/planar_scan.h"

#include <stdint.h>
#include <stdio.h>

static int check_valid_plan(void) {
  const GeozlGpuFrameShape frames[] = {
      {0, 24, 4, 2},
      {24, 12, 3, 1},
  };
  GeozlGpuPlanePlan plan;
  if (geozl_gpu_plane_plan_build(&plan, frames, 2, NULL) != 0)
    return 1;
  const int failed = plan.count != 4 || plan.maxWidth != 4 ||
                     plan.maxRows != 0 || plan.tasks[0].outBase != 0 ||
                     plan.tasks[0].width != 4 || plan.tasks[0].rows != 3 ||
                     plan.tasks[1].outBase != 12 ||
                     plan.tasks[2].outBase != 24 || plan.tasks[2].width != 3 ||
                     plan.tasks[2].rows != 4 || plan.tasks[3].rows != 0;
  if (failed)
    fprintf(stderr,
            "valid plane plan has unexpected tasks: count=%zu width=%u "
            "rows=%u bases=%u,%u,%u pad=%u\n",
            plan.count, plan.maxWidth, plan.maxRows, plan.tasks[0].outBase,
            plan.tasks[1].outBase, plan.tasks[2].outBase, plan.tasks[3].rows);
  geozl_gpu_plane_plan_free(&plan);
  return failed;
}

static int check_invalid_geometry(void) {
  const GeozlGpuFrameShape invalid[] = {
      {0, 0, 4, 1},
      {0, 16, 0, 1},
      {0, 16, 4, 0},
      {0, 17, 4, 1},
      {(size_t)UINT32_MAX + 1, 16, 4, 1},
  };
  for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i) {
    GeozlGpuPlanePlan plan;
    if (geozl_gpu_plane_plan_build(&plan, &invalid[i], 1, NULL) == 0) {
      fprintf(stderr, "invalid geometry %zu was accepted\n", i);
      geozl_gpu_plane_plan_free(&plan);
      return 1;
    }
  }
  return 0;
}

// 262140 rows and 8388480 columns are the largest that need 65535 groups.
static int check_shape_limits(void) {
  const struct {
    GeozlGpuFrameShape frame;
    int fits;
  } cases[] = {
      {{0, 7, 7, 1}, 1},
      {{0, (size_t)1 << 20, 1, (uint32_t)1 << 20}, 0},
      {{0, 2 * 255, 255, 2}, 0},
      {{0, 2 * 256, 256, 2}, 1},
      {{0, 2 * 256, 1, 2}, 1},
      {{0, (size_t)256 * 1048576, 256, 1}, 1},
      {{0, (size_t)257 * 262140, 257, 1}, 1},
      {{0, (size_t)257 * 262141, 257, 1}, 0},
      {{0, 8388480, 8388480, 1}, 1},
      {{0, 8388481, 8388481, 1}, 0},
  };
  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    if (geozl_gpu_plane_shape_fits(&cases[i].frame, 1) != cases[i].fits) {
      fprintf(stderr, "shape %zu took the wrong route\n", i);
      return 1;
    }
  }
  // Metal has no such grid, so only short planes leave the device there.
  const GeozlGpuFrameShape wide = {0, 8388481, 8388481, 1};
  const GeozlGpuFrameShape tall = {0, (size_t)257 * 262141, 257, 1};
  const GeozlGpuFrameShape shortPlanes = {0, 2 * 255, 255, 2};
  if (!geozl_gpu_plane_shape_fits(&wide, 0) ||
      !geozl_gpu_plane_shape_fits(&tall, 0) ||
      geozl_gpu_plane_shape_fits(&shortPlanes, 0)) {
    fprintf(stderr, "a backend with no grid limit took the wrong route\n");
    return 1;
  }
  return 0;
}

int main(void) {
  if (check_valid_plan() || check_invalid_geometry() || check_shape_limits()) {
    fprintf(stderr, "test_planar_scan: failed\n");
    return 1;
  }
  puts("test_planar_scan: ok");
  return 0;
}
