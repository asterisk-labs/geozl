#include "runtime_internal.h"

#include <stdint.h>
#include <string.h>

static const GeozlGpuBackendOps *backend_ops(GeozlGpuBackendId backend) {
  switch (backend) {
#ifdef GEOZL_GPU_METAL
  case GEOZL_GPU_BACKEND_METAL:
    return &geozl_gpu_metal_ops;
#endif
#ifdef GEOZL_GPU_CUDA
  case GEOZL_GPU_BACKEND_CUDA:
    return &geozl_gpu_cuda_ops;
#endif
  default:
    return NULL;
  }
}

int geozl_gpu_backend_available(GeozlGpuBackendId backend) {
  return backend_ops(backend) != NULL;
}

GeozlGpuProgram *geozl_gpu_program_create(GeozlGpuBackendId backend,
                                          const void *module, size_t moduleSize,
                                          int device,
                                          const char *const kernels[],
                                          size_t kernelCount, char *error,
                                          size_t errorCapacity) {
  const GeozlGpuBackendOps *ops = backend_ops(backend);
  if (ops == NULL) {
    geozl_gpu_error(error, errorCapacity, "GPU backend is not built in");
    return NULL;
  }
  if (module == NULL || moduleSize == 0 || device < 0 || kernels == NULL ||
      kernelCount == 0 || kernelCount > GEOZL_GPU_MAX_KERNELS) {
    geozl_gpu_error(error, errorCapacity, "invalid GPU program");
    return NULL;
  }
  for (size_t i = 0; i < kernelCount; ++i) {
    if (kernels[i] == NULL) {
      geozl_gpu_error(error, errorCapacity, "kernel %zu has no name", i);
      return NULL;
    }
  }
  GeozlGpuProgram *program = ops->program_create(
      module, moduleSize, device, kernels, kernelCount, error, errorCapacity);
  if (program != NULL) {
    program->ops = ops;
    program->device = device;
    program->kernelCount = kernelCount;
  }
  return program;
}

void geozl_gpu_program_destroy(GeozlGpuProgram *program) {
  if (program != NULL)
    program->ops->program_destroy(program);
}

const char *geozl_gpu_program_backend(const GeozlGpuProgram *program) {
  return program == NULL ? NULL : program->ops->name;
}

const char *geozl_gpu_program_device(const GeozlGpuProgram *program) {
  return program == NULL ? NULL : program->ops->program_device(program);
}

GeozlGpuStaging *geozl_gpu_staging_create(GeozlGpuProgram *program,
                                          size_t bytes, char *error,
                                          size_t errorCapacity) {
  if (program == NULL || bytes == 0) {
    geozl_gpu_error(error, errorCapacity, "invalid GPU staging");
    return NULL;
  }
  GeozlGpuStaging *staging =
      program->ops->staging_create(program, bytes, error, errorCapacity);
  if (staging != NULL) {
    staging->program = program;
    staging->bytes = bytes;
  }
  return staging;
}

void geozl_gpu_staging_destroy(GeozlGpuStaging *staging) {
  if (staging != NULL)
    staging->program->ops->staging_destroy(staging);
}

void *geozl_gpu_staging_data(const GeozlGpuStaging *staging) {
  return staging == NULL ? NULL : staging->data;
}

size_t geozl_gpu_staging_capacity(const GeozlGpuStaging *staging) {
  return staging == NULL ? 0 : staging->bytes;
}

static int staging_valid(const GeozlGpuProgram *program,
                         const GeozlGpuBuffer *buffer) {
  const GeozlGpuStaging *staging = buffer->staging;
  if (staging == NULL)
    return 1;
  const uintptr_t start = (uintptr_t)staging->data;
  const uintptr_t host = (uintptr_t)buffer->host;
  return staging->program == program && buffer->flags == GEOZL_GPU_COPY_IN &&
         host >= start && host - start <= staging->bytes &&
         (host - start) % GEOZL_GPU_STAGING_ALIGNMENT == 0 &&
         buffer->bytes <= staging->bytes - (host - start);
}

GeozlGpuBatch *geozl_gpu_batch_create(GeozlGpuProgram *program,
                                      const GeozlGpuBuffer buffers[],
                                      size_t bufferCount, char *error,
                                      size_t errorCapacity) {
  if (program == NULL || !geozl_gpu_buffers_valid(buffers, bufferCount)) {
    geozl_gpu_error(error, errorCapacity, "invalid GPU buffers");
    return NULL;
  }
  for (size_t i = 0; i < bufferCount; ++i) {
    if (!staging_valid(program, &buffers[i])) {
      geozl_gpu_error(error, errorCapacity, "GPU buffer %zu is outside staging",
                      i);
      return NULL;
    }
  }
  return program->ops->batch_create(program, buffers, bufferCount, error,
                                    errorCapacity);
}

void geozl_gpu_batch_destroy(GeozlGpuBatch *batch) {
  if (batch != NULL)
    batch->program->ops->batch_destroy(batch);
}

int geozl_gpu_batch_run(GeozlGpuBatch *batch,
                        const GeozlGpuDispatch dispatches[],
                        size_t dispatchCount, GeozlGpuRunTimes *times,
                        char *error, size_t errorCapacity) {
  if (batch == NULL || batch->detached || times == NULL ||
      !geozl_gpu_dispatches_valid(dispatches, dispatchCount,
                                  batch->program->kernelCount))
    return geozl_gpu_error(error, errorCapacity, "invalid GPU dispatches");
  if (batch->program->ops->dispatches_fit(
          batch->program, dispatches, dispatchCount, error, errorCapacity) != 0)
    return 1;
  batch->completed = 0;
  memset(times, 0, sizeof(*times));
  if (batch->program->ops->batch_run(batch, dispatches, dispatchCount, times,
                                     error, errorCapacity) != 0)
    return 1;
  batch->completed = 1;
  return 0;
}

int geozl_gpu_batch_download(GeozlGpuBatch *batch, char *error,
                             size_t errorCapacity) {
  if (batch == NULL || !batch->completed || batch->detached)
    return geozl_gpu_error(error, errorCapacity, "invalid GPU batch");
  return batch->program->ops->batch_download(batch, error, errorCapacity);
}

GeozlGpuMemory *geozl_gpu_batch_take(GeozlGpuBatch *batch, size_t bufferIndex,
                                     char *error, size_t errorCapacity) {
  if (batch == NULL || !batch->completed || batch->detached ||
      bufferIndex >= batch->bufferCount ||
      batch->buffers[bufferIndex].alias >= 0 ||
      batch->buffers[bufferIndex].host != NULL ||
      (batch->buffers[bufferIndex].flags & GEOZL_GPU_RESULT) == 0) {
    geozl_gpu_error(error, errorCapacity,
                    "GPU buffer cannot become an owned result");
    return NULL;
  }
  GeozlGpuMemory *memory =
      batch->program->ops->batch_take(batch, bufferIndex, error, errorCapacity);
  if (memory == NULL)
    return NULL;
  memory->ops = batch->program->ops;
  memory->device = batch->program->device;
  memory->bytes = batch->buffers[bufferIndex].bytes;
  memory->elementBytes = batch->buffers[bufferIndex].elementBytes;
  batch->detached = 1;
  return memory;
}

void geozl_gpu_memory_destroy(GeozlGpuMemory *memory) {
  if (memory != NULL)
    memory->ops->memory_destroy(memory);
}

const char *geozl_gpu_memory_backend(const GeozlGpuMemory *memory) {
  return memory == NULL ? NULL : memory->ops->name;
}

int geozl_gpu_memory_device(const GeozlGpuMemory *memory) {
  return memory == NULL ? -1 : memory->device;
}

void *geozl_gpu_memory_data(const GeozlGpuMemory *memory) {
  return memory == NULL ? NULL : memory->ops->memory_data(memory);
}

size_t geozl_gpu_memory_capacity(const GeozlGpuMemory *memory) {
  return memory == NULL ? 0 : memory->bytes;
}

size_t geozl_gpu_memory_element_size(const GeozlGpuMemory *memory) {
  return memory == NULL ? 0 : memory->elementBytes;
}

int geozl_gpu_memory_download(const GeozlGpuMemory *memory, void *dst,
                              size_t bytes, char *error, size_t errorCapacity) {
  if (memory == NULL || bytes > memory->bytes || (dst == NULL && bytes != 0))
    return geozl_gpu_error(error, errorCapacity, "invalid GPU result download");
  if (bytes == 0)
    return 0;
  return memory->ops->memory_download(memory, dst, bytes, error, errorCapacity);
}
