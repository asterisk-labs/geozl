#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "runtime_internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

@interface GeozlMetalProgram : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, copy) NSArray<id<MTLComputePipelineState>> *pipelines;
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

typedef struct {
  GeozlGpuProgram base;
  void *metal;
  char deviceName[128];
} MetalProgram;

typedef struct {
  GeozlGpuBatch base;
  void *metal;
} MetalBatch;

typedef struct {
  GeozlGpuMemory base;
  void *metal;
} MetalMemory;

static GeozlMetalProgram *metal_program(const GeozlGpuProgram *program) {
  return (__bridge GeozlMetalProgram *)((const MetalProgram *)program)->metal;
}

static GeozlMetalBatch *metal_batch(const GeozlGpuBatch *batch) {
  return (__bridge GeozlMetalBatch *)((const MetalBatch *)batch)->metal;
}

static GeozlMetalMemory *metal_memory(const GeozlGpuMemory *memory) {
  return (__bridge GeozlMetalMemory *)((const MetalMemory *)memory)->metal;
}

static const char *metal_message(NSError *error) {
  const char *message = error.localizedDescription.UTF8String;
  return message == NULL ? "unknown error" : message;
}

// Blits on macOS move whole 4-byte words, so device-only buffers are padded.
static size_t blit_size(size_t bytes) {
  return bytes <= SIZE_MAX - 3 ? (bytes + 3) & ~(size_t)3 : 0;
}

static double seconds_now(void) {
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
}

static GeozlGpuProgram *
metal_program_create(const void *module, size_t moduleSize, int deviceIndex,
                     const char *const kernels[], size_t kernelCount,
                     char *errorBuffer, size_t errorCapacity) {
  if (deviceIndex != 0) {
    geozl_gpu_error(errorBuffer, errorCapacity, "Metal exposes only device 0");
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
      geozl_gpu_error(errorBuffer, errorCapacity,
                      "Metal compilation failed: %s",
                      metal_message(metalError));
      return NULL;
    }

    NSMutableArray<id<MTLComputePipelineState>> *pipelines =
        [NSMutableArray arrayWithCapacity:kernelCount];
    for (size_t i = 0; i < kernelCount; ++i) {
      NSString *name = [NSString stringWithUTF8String:kernels[i]];
      id<MTLFunction> function = [library newFunctionWithName:name];
      if (function == nil) {
        geozl_gpu_error(errorBuffer, errorCapacity, "Metal module lacks %s",
                        kernels[i]);
        return NULL;
      }
      id<MTLComputePipelineState> pipeline =
          [device newComputePipelineStateWithFunction:function
                                                error:&metalError];
      if (pipeline == nil) {
        geozl_gpu_error(errorBuffer, errorCapacity,
                        "no Metal pipeline for %s: %s", kernels[i],
                        metal_message(metalError));
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

    MetalProgram *program = calloc(1, sizeof(*program));
    if (program == NULL) {
      geozl_gpu_error(errorBuffer, errorCapacity, "out of memory");
      return NULL;
    }
    snprintf(program->deviceName, sizeof(program->deviceName), "%s",
             device.name.UTF8String ?: "Metal device");
    program->metal = (__bridge_retained void *)metal;
    return &program->base;
  }
}

static void metal_program_destroy(GeozlGpuProgram *base) {
  MetalProgram *program = (MetalProgram *)base;
  CFBridgingRelease(program->metal);
  free(program);
}

static const char *metal_program_device(const GeozlGpuProgram *program) {
  return ((const MetalProgram *)program)->deviceName;
}

static GeozlGpuBatch *metal_batch_create(GeozlGpuProgram *program,
                                         const GeozlGpuBuffer buffers[],
                                         size_t bufferCount, char *errorBuffer,
                                         size_t errorCapacity) {
  @autoreleasepool {
    GeozlMetalProgram *metalProgram = metal_program(program);
    NSMutableArray<id<MTLBuffer>> *metalBuffers =
        [NSMutableArray arrayWithCapacity:bufferCount];
    const long systemPageSize = sysconf(_SC_PAGESIZE);
    const size_t pageSize = systemPageSize > 0 ? (size_t)systemPageSize : 4096;
    for (size_t i = 0; i < bufferCount; ++i) {
      const GeozlGpuBuffer *buffer = &buffers[i];
      id<MTLBuffer> metalBuffer = nil;
      if (buffer->alias >= 0) {
        metalBuffer = metalBuffers[(NSUInteger)buffer->alias];
      } else if (buffer->host == NULL) {
        const size_t length = blit_size(buffer->bytes);
        metalBuffer =
            length == 0
                ? nil
                : [metalProgram.device
                      newBufferWithLength:length
                                  options:MTLResourceStorageModePrivate];
      } else if ((uintptr_t)buffer->host % pageSize == 0 &&
                 buffer->bytes % pageSize == 0) {
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
    MetalBatch *batch = calloc(1, sizeof(*batch));
    if (batch == NULL) {
      geozl_gpu_error(errorBuffer, errorCapacity, "out of memory");
      return NULL;
    }
    geozl_gpu_batch_init(&batch->base, program, buffers, bufferCount);
    batch->metal = (__bridge_retained void *)metal;
    return &batch->base;
  }
}

static void metal_batch_destroy(GeozlGpuBatch *base) {
  MetalBatch *batch = (MetalBatch *)base;
  CFBridgingRelease(batch->metal);
  free(batch);
}

static void wait_all(id<MTLCommandBuffer> __strong *commands, size_t count) {
  for (size_t i = 0; i < count; ++i)
    [commands[i] waitUntilCompleted];
}

static int metal_dispatches_fit(const GeozlGpuProgram *program,
                                const GeozlGpuDispatch dispatches[],
                                size_t dispatchCount, char *errorBuffer,
                                size_t errorCapacity) {
  @autoreleasepool {
    NSArray<id<MTLComputePipelineState>> *pipelines =
        metal_program(program).pipelines;
    for (size_t i = 0; i < dispatchCount; ++i) {
      if (dispatches[i].threads >
          pipelines[dispatches[i].kernel].maxTotalThreadsPerThreadgroup)
        return geozl_gpu_error(errorBuffer, errorCapacity,
                               "Metal dispatch %zu requests too many threads",
                               i);
    }
    return 0;
  }
}

static int metal_batch_run(GeozlGpuBatch *batch,
                           const GeozlGpuDispatch dispatches[],
                           size_t dispatchCount, GeozlGpuRunTimes *times,
                           char *errorBuffer, size_t errorCapacity) {
  @autoreleasepool {
    GeozlMetalBatch *metal = metal_batch(batch);
    const double uploadStart = seconds_now();
    for (size_t i = 0; i < batch->bufferCount; ++i) {
      const GeozlGpuBuffer *buffer = &batch->buffers[i];
      id<MTLBuffer> metalBuffer = metal.buffers[i];
      if (buffer->alias < 0 && (buffer->flags & GEOZL_GPU_COPY_IN) != 0 &&
          metalBuffer.contents != buffer->host)
        memcpy(metalBuffer.contents, buffer->host, buffer->bytes);
    }
    times->upload = seconds_now() - uploadStart;

    id<MTLCommandBuffer> commands[GEOZL_GPU_MAX_KERNELS];
    for (size_t i = 0; i < dispatchCount; ++i) {
      const GeozlGpuDispatch *dispatch = &dispatches[i];
      commands[i] = [metal.program.queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder =
          [commands[i] computeCommandEncoder];
      if (commands[i] == nil || encoder == nil) {
        wait_all(commands, i);
        return geozl_gpu_error(errorBuffer, errorCapacity,
                               "Metal command allocation failed");
      }
      [encoder
          setComputePipelineState:metal.program.pipelines[dispatch->kernel]];
      for (size_t b = 0; b < batch->bufferCount; ++b)
        [encoder setBuffer:metal.buffers[b] offset:0 atIndex:b];
      [encoder dispatchThreadgroups:MTLSizeMake(dispatch->groupsX,
                                                dispatch->groupsY, 1)
              threadsPerThreadgroup:MTLSizeMake(dispatch->threads, 1, 1)];
      [encoder endEncoding];
      [commands[i] commit];
    }

    wait_all(commands, dispatchCount);
    for (size_t i = 0; i < dispatchCount; ++i) {
      if (commands[i].status != MTLCommandBufferStatusCompleted)
        return geozl_gpu_error(errorBuffer, errorCapacity,
                               "Metal kernel %zu failed: %s", i,
                               metal_message(commands[i].error));
      times->dispatch[i] = commands[i].GPUEndTime - commands[i].GPUStartTime;
    }
    return 0;
  }
}

static int metal_batch_download(GeozlGpuBatch *batch, char *errorBuffer,
                                size_t errorCapacity) {
  (void)errorBuffer;
  (void)errorCapacity;
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

static GeozlGpuMemory *metal_batch_take(GeozlGpuBatch *batch,
                                        size_t bufferIndex, char *errorBuffer,
                                        size_t errorCapacity) {
  @autoreleasepool {
    GeozlMetalBatch *metalBatch = metal_batch(batch);
    GeozlMetalMemory *metalMemory = [GeozlMetalMemory new];
    metalMemory.device = metalBatch.program.device;
    metalMemory.queue = metalBatch.program.queue;
    metalMemory.buffer = metalBatch.buffers[bufferIndex];

    MetalMemory *memory = calloc(1, sizeof(*memory));
    if (memory == NULL) {
      geozl_gpu_error(errorBuffer, errorCapacity, "out of memory");
      return NULL;
    }
    memory->metal = (__bridge_retained void *)metalMemory;
    return &memory->base;
  }
}

static void metal_memory_destroy(GeozlGpuMemory *base) {
  MetalMemory *memory = (MetalMemory *)base;
  CFBridgingRelease(memory->metal);
  free(memory);
}

static void *metal_memory_data(const GeozlGpuMemory *memory) {
  return (__bridge void *)metal_memory(memory).buffer;
}

static int metal_memory_download(const GeozlGpuMemory *memory, void *dst,
                                 size_t bytes, char *errorBuffer,
                                 size_t errorCapacity) {
  @autoreleasepool {
    GeozlMetalMemory *metal = metal_memory(memory);
    const size_t copyBytes = blit_size(bytes);
    if (copyBytes == 0 || copyBytes > metal.buffer.length)
      return geozl_gpu_error(errorBuffer, errorCapacity,
                             "invalid Metal result download");
    id<MTLBuffer> staging =
        [metal.device newBufferWithLength:copyBytes
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
                     size:copyBytes];
    [blit endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted)
      return geozl_gpu_error(errorBuffer, errorCapacity,
                             "Metal result download failed: %s",
                             metal_message(command.error));
    memcpy(dst, staging.contents, bytes);
    return 0;
  }
}

const GeozlGpuBackendOps geozl_gpu_metal_ops = {
    .name = "metal",
    .program_create = metal_program_create,
    .program_destroy = metal_program_destroy,
    .program_device = metal_program_device,
    .batch_create = metal_batch_create,
    .batch_destroy = metal_batch_destroy,
    .dispatches_fit = metal_dispatches_fit,
    .batch_run = metal_batch_run,
    .batch_download = metal_batch_download,
    .batch_take = metal_batch_take,
    .memory_destroy = metal_memory_destroy,
    .memory_data = metal_memory_data,
    .memory_download = metal_memory_download,
};
