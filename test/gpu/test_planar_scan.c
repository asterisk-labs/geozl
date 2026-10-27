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
                     plan.maxRows != 4 || plan.tasks[0].outBase != 0 ||
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

int main(void) {
  if (check_valid_plan() || check_invalid_geometry()) {
    fprintf(stderr, "test_planar_scan: failed\n");
    return 1;
  }
  puts("test_planar_scan: ok");
  return 0;
}
