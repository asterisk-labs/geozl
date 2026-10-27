#ifndef GEOZL_GPU_RUNTIME_H
#define GEOZL_GPU_RUNTIME_H

#include <stddef.h>

#define GEOZL_GPU_MAX_BUFFERS 16
#define GEOZL_GPU_MAX_KERNELS 8

enum {
  GEOZL_GPU_COPY_IN = 1u,
  GEOZL_GPU_COPY_OUT = 2u,
  GEOZL_GPU_RESULT = 4u,
};

// Each backend compiles in only when its GEOZL_GPU_<NAME> macro is defined.
// New backends append here; the values are not stored anywhere.
typedef enum {
  GEOZL_GPU_BACKEND_METAL,
  GEOZL_GPU_BACKEND_CUDA,
} GeozlGpuBackendId;

typedef struct GeozlGpuProgram GeozlGpuProgram;
typedef struct GeozlGpuBatch GeozlGpuBatch;
typedef struct GeozlGpuMemory GeozlGpuMemory;

typedef struct {
  // host is required by COPY_IN/COPY_OUT and otherwise must be NULL. RESULT
  // marks a device allocation that can be moved out with batch_take.
  void *host;
  size_t bytes;
  size_t elementBytes;
  unsigned flags;
  int alias;
} GeozlGpuBuffer;

typedef struct {
  unsigned kernel;
  unsigned groupsX;
  unsigned groupsY;
  unsigned threads;
} GeozlGpuDispatch;

typedef struct {
  double upload;
  double dispatch[GEOZL_GPU_MAX_KERNELS];
} GeozlGpuRunTimes;

// Programs own the backend device and compiled kernels. Batches own their
// device buffers and must be destroyed before their program. A program and its
// batches may be reused between calls, but are not safe to run concurrently.

int geozl_gpu_backend_available(GeozlGpuBackendId backend);

// module is Metal source for METAL and PTX for CUDA, compiled by Slang from
// gpu/kernels. Returns NULL when the backend is not built in.
GeozlGpuProgram *geozl_gpu_program_create(GeozlGpuBackendId backend,
                                          const void *module, size_t moduleSize,
                                          int device,
                                          const char *const kernels[],
                                          size_t kernelCount, char *error,
                                          size_t errorCapacity);
void geozl_gpu_program_destroy(GeozlGpuProgram *program);

const char *geozl_gpu_program_backend(const GeozlGpuProgram *program);
const char *geozl_gpu_program_device(const GeozlGpuProgram *program);

GeozlGpuBatch *geozl_gpu_batch_create(GeozlGpuProgram *program,
                                      const GeozlGpuBuffer buffers[],
                                      size_t bufferCount, char *error,
                                      size_t errorCapacity);
void geozl_gpu_batch_destroy(GeozlGpuBatch *batch);

int geozl_gpu_batch_run(GeozlGpuBatch *batch,
                        const GeozlGpuDispatch dispatches[],
                        size_t dispatchCount, GeozlGpuRunTimes *times,
                        char *error, size_t errorCapacity);

// Copies buffers marked GEOZL_GPU_COPY_OUT to their host pointers. A caller
// that wants to keep the result on the device simply does not call this.
int geozl_gpu_batch_download(GeozlGpuBatch *batch, char *error,
                             size_t errorCapacity);

// Transfers ownership of a device allocation out of the batch. The buffer must
// be marked GEOZL_GPU_RESULT, must not alias another buffer and must not use
// caller-owned storage. Taking a result ends the batch: it can only be
// destroyed afterwards.
GeozlGpuMemory *geozl_gpu_batch_take(GeozlGpuBatch *batch, size_t buffer,
                                     char *error, size_t errorCapacity);

void geozl_gpu_memory_destroy(GeozlGpuMemory *memory);
const char *geozl_gpu_memory_backend(const GeozlGpuMemory *memory);
int geozl_gpu_memory_device(const GeozlGpuMemory *memory);
void *geozl_gpu_memory_data(const GeozlGpuMemory *memory);
// Capacity is the backing allocation, which may include decoder scratch. Use
// the byte count returned by the decoder for the logical result size.
size_t geozl_gpu_memory_capacity(const GeozlGpuMemory *memory);
size_t geozl_gpu_memory_element_size(const GeozlGpuMemory *memory);
int geozl_gpu_memory_download(const GeozlGpuMemory *memory, void *dst,
                              size_t bytes, char *error, size_t errorCapacity);

#endif // GEOZL_GPU_RUNTIME_H
