// A small host layer over Metal and the CUDA driver API: bind buffers in the
// order the Slang module declares them, run a list of kernels, time them.
// It stands in for slang-rhi in the prototype.

#ifndef GEOZL_GPU_PROTO_GPU_H
#define GEOZL_GPU_PROTO_GPU_H

#include <stddef.h>

#define GPU_MAX_BUFFERS 16
#define GPU_MAX_DISPATCHES 8

typedef struct {
  void *host; // NULL for device-only scratch
  size_t bytes;
  // Bytes per element: 1 for byte-address buffers, sizeof(T) for structured
  // buffers. The CUDA target records the element count beside each pointer.
  size_t elem;
  int output;  // copied back to host after the run
  int aliasOf; // -1, or the index of an earlier buffer this one views
} GpuBuffer;

typedef struct {
  const char *entry;
  unsigned groupsX, groupsY;
  unsigned threads;
} GpuDispatch;

typedef struct {
  double kernel; // best total of all dispatches
  double
      endToEnd; // best submit-to-device-ready wall time, including input copy
  double copy;  // best host to device copy (0 with unified memory)
  double pass[GPU_MAX_DISPATCHES]; // best time of each dispatch
  char device[128];
} GpuTimes;

// module is MSL source for Metal and PTX for CUDA.
int gpu_run(const char *module, GpuBuffer *buffers, int nBuffers,
            const GpuDispatch *dispatches, int nDispatches, int reps,
            GpuTimes *times);

extern const char *gpu_backend_name;

#endif // GEOZL_GPU_PROTO_GPU_H
