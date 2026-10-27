#include "gpu.h"

#include "gpu/runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *gpu_backend_name = "gpu";

double now_seconds(void);

void *gpu_load_module(const char *path, size_t *size) {
  FILE *file = fopen(path, "rb");
  if (file == NULL || fseek(file, 0, SEEK_END) != 0) {
    if (file != NULL)
      fclose(file);
    return NULL;
  }
  const long length = ftell(file);
  if (length <= 0 || fseek(file, 0, SEEK_SET) != 0) {
    fclose(file);
    return NULL;
  }
  void *data = malloc((size_t)length);
  if (data == NULL || fread(data, 1, (size_t)length, file) != (size_t)length) {
    free(data);
    fclose(file);
    return NULL;
  }
  fclose(file);
  *size = (size_t)length;
  return data;
}

int gpu_run(const char *modulePath, GpuBuffer *buffers, int bufferCount,
            const GpuDispatch *dispatches, int dispatchCount, int reps,
            GpuTimes *times) {
  if (modulePath == NULL || buffers == NULL || dispatches == NULL ||
      times == NULL || reps <= 0 || bufferCount <= 0 ||
      bufferCount > GPU_MAX_BUFFERS || dispatchCount <= 0 ||
      dispatchCount > GPU_MAX_DISPATCHES)
    return 1;

  size_t moduleSize = 0;
  void *module = gpu_load_module(modulePath, &moduleSize);
  if (module == NULL) {
    fprintf(stderr, "cannot read %s\n", modulePath);
    return 1;
  }

  const char *kernels[GPU_MAX_DISPATCHES];
  GeozlGpuDispatch gpuDispatches[GPU_MAX_DISPATCHES];
  for (int i = 0; i < dispatchCount; ++i) {
    kernels[i] = dispatches[i].entry;
    gpuDispatches[i] =
        (GeozlGpuDispatch){(unsigned)i, dispatches[i].groupsX,
                           dispatches[i].groupsY, dispatches[i].threads};
  }
  const char *deviceEnv = getenv("GPU");
  const int device = deviceEnv == NULL ? 0 : atoi(deviceEnv);
  char error[512];
  GeozlGpuProgram *program =
      geozl_gpu_program_create(GPU_BACKEND, module, moduleSize, device, kernels,
                               (size_t)dispatchCount, error, sizeof(error));
  free(module);
  if (program == NULL) {
    fprintf(stderr, "%s\n", error);
    return 1;
  }

  GeozlGpuBuffer gpuBuffers[GPU_MAX_BUFFERS];
  for (int i = 0; i < bufferCount; ++i) {
    unsigned flags = 0;
    if (buffers[i].aliasOf < 0) {
      if (buffers[i].output)
        flags = GEOZL_GPU_COPY_OUT;
      else if (buffers[i].host != NULL)
        flags = GEOZL_GPU_COPY_IN;
    }
    gpuBuffers[i] =
        (GeozlGpuBuffer){buffers[i].host, buffers[i].bytes, buffers[i].elem,
                         flags, buffers[i].aliasOf};
  }
  GeozlGpuBatch *batch = geozl_gpu_batch_create(
      program, gpuBuffers, (size_t)bufferCount, error, sizeof(error));
  if (batch == NULL) {
    fprintf(stderr, "%s\n", error);
    geozl_gpu_program_destroy(program);
    return 1;
  }

  times->kernel = times->endToEnd = times->copy = 1e30;
  for (int i = 0; i < dispatchCount; ++i)
    times->pass[i] = 1e30;
  for (int r = 0; r < reps; ++r) {
    GeozlGpuRunTimes runTimes;
    const double start = now_seconds();
    if (geozl_gpu_batch_run(batch, gpuDispatches, (size_t)dispatchCount,
                            &runTimes, error, sizeof(error)) != 0) {
      fprintf(stderr, "%s\n", error);
      geozl_gpu_batch_destroy(batch);
      geozl_gpu_program_destroy(program);
      return 1;
    }
    const double wall = now_seconds() - start;
    double kernel = 0;
    for (int i = 0; i < dispatchCount; ++i) {
      kernel += runTimes.dispatch[i];
      if (runTimes.dispatch[i] < times->pass[i])
        times->pass[i] = runTimes.dispatch[i];
    }
    if (runTimes.upload < times->copy)
      times->copy = runTimes.upload;
    if (kernel < times->kernel)
      times->kernel = kernel;
    if (wall < times->endToEnd)
      times->endToEnd = wall;
  }

  int failed = geozl_gpu_batch_download(batch, error, sizeof(error));
  if (failed)
    fprintf(stderr, "%s\n", error);
  snprintf(times->device, sizeof(times->device), "%s",
           geozl_gpu_program_device(program));
  gpu_backend_name = geozl_gpu_program_backend(program);
  geozl_gpu_batch_destroy(batch);
  geozl_gpu_program_destroy(program);
  return failed;
}
