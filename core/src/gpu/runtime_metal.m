#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "runtime_internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

@interface GeozlMetalProgram : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, copy) NSArray<id<MTLComputePipelineState>> *pipelines;
@property(nonatomic, copy) NSString *deviceName;
@end

@implementation GeozlMetalProgram
@end

@interface GeozlMetalBatch : NSObject
@property(nonatomic, strong) GeozlMetalProgram *program;
@property(nonatomic, copy) NSArray<id<MTLBuffer>> *buffers;
@end

@implementation GeozlMetalBatch
@end

@interface GeozlMetalMemory : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLBuffer> buffer;
@end

@implementation GeozlMetalMemory
@end

struct GeozlGpuProgram {
  void *metal;
  size_t kernelCount;
};

struct GeozlGpuBatch {
  GeozlGpuProgram *program;
  void *metal;
  GeozlGpuBuffer buffers[GEOZL_GPU_MAX_BUFFERS];
  unsigned char external[GEOZL_GPU_MAX_BUFFERS];
  size_t bufferCount;
  int completed;
  int detached;
};

struct GeozlGpuMemory {
  void *metal;
  size_t bytes;
  size_t elementBytes;
};

static GeozlMetalProgram *metal_program(const GeozlGpuProgram *program) {
  return (__bridge GeozlMetalProgram *)program->metal;
}

static GeozlMetalBatch *metal_batch(const GeozlGpuBatch *batch) {
  return (__bridge GeozlMetalBatch *)batch->metal;
}

static GeozlMetalMemory *metal_memory(const GeozlGpuMemory *memory) {
  return (__bridge GeozlMetalMemory *)memory->metal;
}

GeozlGpuProgram *geozl_gpu_program_create(const void *module, size_t moduleSize,
                                          int deviceIndex,
                                          const char *const kernels[],
                                          size_t kernelCount, char *errorBuffer,
                                          size_t errorCapacity) {
  if (module == NULL || moduleSize == 0 || deviceIndex != 0 ||
      kernels == NULL || kernelCount == 0 ||
      kernelCount > GEOZL_GPU_MAX_KERNELS) {
    geozl_gpu_error(errorBuffer, errorCapacity, "invalid Metal program");
    return NULL;
  }

  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      geozl_gpu_error(errorBuffer, errorCapacity, "Metal device unavailable");
      return NULL;
    }

    NSString *source = [[NSString alloc] initWithBytes:module
                                                length:moduleSize
                                              encoding:NSUTF8StringEncoding];
    if (source == nil) {
      geozl_gpu_error(errorBuffer, errorCapacity,
                      "Metal module is not valid UTF-8");
      return NULL;
    }
    NSError *metalError = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:source
                                                  options:nil
                                                    error:&metalError];
    if (library == nil) {
      geozl_gpu_error(
          errorBuffer, errorCapacity, "Metal compilation failed: %s",
          metalError.localizedDescription.UTF8String ?: "unknown error");
      return NULL;
    }

    NSMutableArray<id<MTLComputePipelineState>> *pipelines =
        [NSMutableArray arrayWithCapacity:kernelCount];
    for (size_t i = 0; i < kernelCount; ++i) {
      if (kernels[i] == NULL) {
        geozl_gpu_error(errorBuffer, errorCapacity, "kernel %zu has no name",
                        i);
        return NULL;
      }
      NSString *name = [NSString stringWithUTF8String:kernels[i]];
      id<MTLFunction> function = [library newFunctionWithName:name];
      id<MTLComputePipelineState> pipeline =
          function == nil
              ? nil
              : [device newComputePipelineStateWithFunction:function
                                                      error:&metalError];
      if (pipeline == nil) {
        geozl_gpu_error(errorBuffer, errorCapacity, "no Metal pipeline for %s",
                        kernels[i]);
        return NULL;
      }
      if (pipeline.threadExecutionWidth != 32) {
        geozl_gpu_error(errorBuffer, errorCapacity,
                        "%s needs a 32-wide SIMD group", kernels[i]);
        return NULL;
      }
      [pipelines addObject:pipeline];
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) {
      geozl_gpu_error(errorBuffer, errorCapacity,
                      "Metal command queue allocation failed");
      return NULL;
    }

    GeozlMetalProgram *metal = [GeozlMetalProgram new];
    metal.device = device;
    metal.queue = queue;
    metal.pipelines = pipelines;
    metal.deviceName = device.name;

    GeozlGpuProgram *program = calloc(1, sizeof(*program));
    if (program == NULL) {
      geozl_gpu_error(errorBuffer, errorCapacity, "out of memory");
      return NULL;
    }
    program->metal = (__bridge_retained void *)metal;
    program->kernelCount = kernelCount;
    return program;
  }
}

void geozl_gpu_program_destroy(GeozlGpuProgram *program) {
  if (program == NULL)
    return;
  CFBridgingRelease(program->metal);
  free(program);
}

const char *geozl_gpu_program_backend(const GeozlGpuProgram *program) {
  return program == NULL ? NULL : "metal";
}

const char *geozl_gpu_program_device(const GeozlGpuProgram *program) {
  return program == NULL ? NULL : metal_program(program).deviceName.UTF8String;
}

GeozlGpuBatch *geozl_gpu_batch_create(GeozlGpuProgram *program,
                                      const GeozlGpuBuffer buffers[],
                                      size_t bufferCount, char *errorBuffer,
                                      size_t errorCapacity) {
  if (program == NULL || !geozl_gpu_buffers_valid(buffers, bufferCount)) {
    geozl_gpu_error(errorBuffer, errorCapacity, "invalid Metal buffers");
    return NULL;
  }

  @autoreleasepool {
    GeozlMetalProgram *metalProgram = metal_program(program);
    NSMutableArray<id<MTLBuffer>> *metalBuffers =
        [NSMutableArray arrayWithCapacity:bufferCount];
    unsigned char external[GEOZL_GPU_MAX_BUFFERS] = {0};
    const long systemPageSize = sysconf(_SC_PAGESIZE);
    const size_t pageSize = systemPageSize > 0 ? (size_t)systemPageSize : 4096;
    for (size_t i = 0; i < bufferCount; ++i) {
      const GeozlGpuBuffer *buffer = &buffers[i];
      id<MTLBuffer> metalBuffer = nil;
      if (buffer->alias >= 0) {
        metalBuffer = metalBuffers[(NSUInteger)buffer->alias];
      } else if (buffer->host == NULL) {
        metalBuffer = [metalProgram.device
            newBufferWithLength:buffer->bytes
                        options:MTLResourceStorageModePrivate];
      } else if ((uintptr_t)buffer->host % pageSize == 0 &&
                 buffer->bytes % pageSize == 0) {
        external[i] = 1;
        metalBuffer = [metalProgram.device
            newBufferWithBytesNoCopy:buffer->host
                              length:buffer->bytes
                             options:MTLResourceStorageModeShared
                         deallocator:nil];
      } else {
        metalBuffer = [metalProgram.device
            newBufferWithLength:buffer->bytes
                        options:MTLResourceStorageModeShared];
      }
      if (metalBuffer == nil) {
        geozl_gpu_error(errorBuffer, errorCapacity,
                        "Metal buffer %zu allocation failed", i);
        return NULL;
      }
      [metalBuffers addObject:metalBuffer];
    }

    GeozlMetalBatch *metal = [GeozlMetalBatch new];
    metal.program = metalProgram;
    metal.buffers = metalBuffers;
    GeozlGpuBatch *batch = calloc(1, sizeof(*batch));
    if (batch == NULL) {
      geozl_gpu_error(errorBuffer, errorCapacity, "out of memory");
      return NULL;
    }
    batch->program = program;
    batch->metal = (__bridge_retained void *)metal;
    memcpy(batch->buffers, buffers, bufferCount * sizeof(*buffers));
    memcpy(batch->external, external, bufferCount * sizeof(*external));
    batch->bufferCount = bufferCount;
    return batch;
  }
}

void geozl_gpu_batch_destroy(GeozlGpuBatch *batch) {
  if (batch == NULL)
    return;
  CFBridgingRelease(batch->metal);
  free(batch);
}

int geozl_gpu_batch_run(GeozlGpuBatch *batch,
                        const GeozlGpuDispatch dispatches[],
                        size_t dispatchCount, GeozlGpuRunTimes *times,
                        char *errorBuffer, size_t errorCapacity) {
  if (batch == NULL || batch->detached || times == NULL ||
      !geozl_gpu_dispatches_valid(dispatches, dispatchCount,
                                  batch == NULL ? 0
                                                : batch->program->kernelCount))
    return geozl_gpu_error(errorBuffer, errorCapacity,
                           "invalid Metal dispatches");
  batch->completed = 0;

  @autoreleasepool {
    GeozlMetalBatch *metal = metal_batch(batch);
    memset(times, 0, sizeof(*times));
    for (size_t i = 0; i < batch->bufferCount; ++i) {
      const GeozlGpuBuffer *buffer = &batch->buffers[i];
      id<MTLBuffer> metalBuffer = metal.buffers[i];
      if (buffer->alias < 0 && (buffer->flags & GEOZL_GPU_COPY_IN) != 0 &&
          metalBuffer.contents != buffer->host)
        memcpy(metalBuffer.contents, buffer->host, buffer->bytes);
    }

    id<MTLCommandBuffer> commands[GEOZL_GPU_MAX_KERNELS];
    for (size_t i = 0; i < dispatchCount; ++i) {
      const GeozlGpuDispatch *dispatch = &dispatches[i];
      id<MTLComputePipelineState> pipeline =
          metal.program.pipelines[dispatch->kernel];
      if (dispatch->threads > pipeline.maxTotalThreadsPerThreadgroup)
        return geozl_gpu_error(errorBuffer, errorCapacity,
                               "kernel %u requests too many threads",
                               dispatch->kernel);
      commands[i] = [metal.program.queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder =
          [commands[i] computeCommandEncoder];
      if (commands[i] == nil || encoder == nil)
        return geozl_gpu_error(errorBuffer, errorCapacity,
                               "Metal command allocation failed");
      [encoder setComputePipelineState:pipeline];
      for (size_t b = 0; b < batch->bufferCount; ++b)
        [encoder setBuffer:metal.buffers[b] offset:0 atIndex:b];
      [encoder dispatchThreadgroups:MTLSizeMake(dispatch->groupsX,
                                                dispatch->groupsY, 1)
              threadsPerThreadgroup:MTLSizeMake(dispatch->threads, 1, 1)];
      [encoder endEncoding];
      [commands[i] commit];
    }

    [commands[dispatchCount - 1] waitUntilCompleted];
    for (size_t i = 0; i < dispatchCount; ++i) {
      [commands[i] waitUntilCompleted];
      if (commands[i].status != MTLCommandBufferStatusCompleted) {
        const char *message = commands[i].error.localizedDescription.UTF8String;
        return geozl_gpu_error(errorBuffer, errorCapacity,
                               "Metal kernel %zu failed: %s", i,
                               message ?: "unknown error");
      }
      times->dispatch[i] = commands[i].GPUEndTime - commands[i].GPUStartTime;
    }
    batch->completed = 1;
    return 0;
  }
}

int geozl_gpu_batch_download(GeozlGpuBatch *batch, char *errorBuffer,
                             size_t errorCapacity) {
  if (batch == NULL || !batch->completed || batch->detached)
    return geozl_gpu_error(errorBuffer, errorCapacity, "invalid Metal batch");
  @autoreleasepool {
    GeozlMetalBatch *metal = metal_batch(batch);
    for (size_t i = 0; i < batch->bufferCount; ++i) {
      const GeozlGpuBuffer *buffer = &batch->buffers[i];
      id<MTLBuffer> metalBuffer = metal.buffers[i];
      if (buffer->alias < 0 && (buffer->flags & GEOZL_GPU_COPY_OUT) != 0 &&
          metalBuffer.contents != buffer->host)
        memcpy(buffer->host, metalBuffer.contents, buffer->bytes);
    }
    return 0;
  }
}

GeozlGpuMemory *geozl_gpu_batch_take(GeozlGpuBatch *batch, size_t bufferIndex,
                                     char *errorBuffer, size_t errorCapacity) {
  if (batch == NULL || !batch->completed || batch->detached ||
      bufferIndex >= batch->bufferCount ||
      batch->buffers[bufferIndex].alias >= 0 ||
      (batch->buffers[bufferIndex].flags & GEOZL_GPU_RESULT) == 0 ||
      batch->external[bufferIndex]) {
    geozl_gpu_error(errorBuffer, errorCapacity,
                    "Metal buffer cannot become an owned result");
    return NULL;
  }

  @autoreleasepool {
    GeozlMetalBatch *metalBatch = metal_batch(batch);
    GeozlMetalMemory *metalMemory = [GeozlMetalMemory new];
    metalMemory.device = metalBatch.program.device;
    metalMemory.queue = metalBatch.program.queue;
    metalMemory.buffer = metalBatch.buffers[bufferIndex];

    GeozlGpuMemory *memory = calloc(1, sizeof(*memory));
    if (memory == NULL) {
      geozl_gpu_error(errorBuffer, errorCapacity, "out of memory");
      return NULL;
    }
    memory->metal = (__bridge_retained void *)metalMemory;
    memory->bytes = batch->buffers[bufferIndex].bytes;
    memory->elementBytes = batch->buffers[bufferIndex].elementBytes;
    batch->detached = 1;
    return memory;
  }
}

void geozl_gpu_memory_destroy(GeozlGpuMemory *memory) {
  if (memory == NULL)
    return;
  CFBridgingRelease(memory->metal);
  free(memory);
}

const char *geozl_gpu_memory_backend(const GeozlGpuMemory *memory) {
  return memory == NULL ? NULL : "metal";
}

int geozl_gpu_memory_device(const GeozlGpuMemory *memory) {
  return memory == NULL ? -1 : 0;
}

void *geozl_gpu_memory_data(const GeozlGpuMemory *memory) {
  return memory == NULL ? NULL : (__bridge void *)metal_memory(memory).buffer;
}

size_t geozl_gpu_memory_capacity(const GeozlGpuMemory *memory) {
  return memory == NULL ? 0 : memory->bytes;
}

size_t geozl_gpu_memory_element_size(const GeozlGpuMemory *memory) {
  return memory == NULL ? 0 : memory->elementBytes;
}

int geozl_gpu_memory_download(const GeozlGpuMemory *memory, void *dst,
                              size_t bytes, char *errorBuffer,
                              size_t errorCapacity) {
  if (memory == NULL || bytes > memory->bytes || (dst == NULL && bytes != 0))
    return geozl_gpu_error(errorBuffer, errorCapacity,
                           "invalid Metal result download");
  if (bytes == 0)
    return 0;

  @autoreleasepool {
    GeozlMetalMemory *metal = metal_memory(memory);
    id<MTLBuffer> staging =
        [metal.device newBufferWithLength:bytes
                                  options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> command = [metal.queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    if (staging == nil || command == nil || blit == nil)
      return geozl_gpu_error(errorBuffer, errorCapacity,
                             "Metal result download allocation failed");
    [blit copyFromBuffer:metal.buffer
             sourceOffset:0
                 toBuffer:staging
        destinationOffset:0
                     size:bytes];
    [blit endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
      const char *message = command.error.localizedDescription.UTF8String;
      return geozl_gpu_error(errorBuffer, errorCapacity,
                             "Metal result download failed: %s",
                             message ?: "unknown error");
    }
    memcpy(dst, staging.contents, bytes);
    return 0;
  }
}
