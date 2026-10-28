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
// Private buffers kept from finished batches for the next ones.
@property(nonatomic, strong) NSMutableArray<id<MTLBuffer>> *spares;
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
  size_t offsets[GEOZL_GPU_MAX_BUFFERS];
  int spare[GEOZL_GPU_MAX_BUFFERS];
} MetalBatch;

typedef struct {
  GeozlGpuMemory base;
  void *metal;
} MetalMemory;

typedef struct {
  GeozlGpuStaging base;
  void *metal;
} MetalStaging;

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

#define METAL_SPARES 4

// A spare serves a request at most four times smaller.
static id<MTLBuffer> acquire_private(GeozlMetalProgram *program,
                                     size_t length) {
  NSUInteger best = NSNotFound;
  for (NSUInteger i = 0; i < program.spares.count; ++i) {
    const size_t spare = program.spares[i].length;
    if (spare >= length &&
        (spare / 4 <= length || spare <= ((size_t)1 << 20)) &&
        (best == NSNotFound || spare < program.spares[best].length))
      best = i;
  }
  if (best == NSNotFound)
    return [program.device newBufferWithLength:length
                                       options:MTLResourceStorageModePrivate];
  id<MTLBuffer> buffer = program.spares[best];
  [program.spares removeObjectAtIndex:best];
  return buffer;
}

// Keeps the largest buffers.
static void release_private(GeozlMetalProgram *program, id<MTLBuffer> buffer) {
  if (program.spares.count < METAL_SPARES) {
    [program.spares addObject:buffer];
    return;
  }
  NSUInteger smallest = 0;
  for (NSUInteger i = 1; i < program.spares.count; ++i) {
    if (program.spares[i].length < program.spares[smallest].length)
      smallest = i;
  }
  if (program.spares[smallest].length < buffer.length)
    program.spares[smallest] = buffer;
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
    metal.spares = [NSMutableArray array];

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

static id<MTLBuffer> staging_buffer(const GeozlGpuStaging *staging) {
  return (__bridge id<MTLBuffer>)((const MetalStaging *)staging)->metal;
}

static GeozlGpuStaging *metal_staging_create(GeozlGpuProgram *program,
                                             size_t bytes, char *errorBuffer,
                                             size_t errorCapacity) {
  @autoreleasepool {
    id<MTLBuffer> buffer = [metal_program(program).device
        newBufferWithLength:bytes
                    options:MTLResourceStorageModeShared];
    MetalStaging *staging = buffer == nil ? NULL : calloc(1, sizeof(*staging));
    if (staging == NULL) {
      geozl_gpu_error(errorBuffer, errorCapacity,
                      "Metal staging allocation failed");
      return NULL;
    }
    staging->base.data = buffer.contents;
    staging->metal = (__bridge_retained void *)buffer;
    return &staging->base;
  }
}

static void metal_staging_destroy(GeozlGpuStaging *base) {
  MetalStaging *staging = (MetalStaging *)base;
  CFBridgingRelease(staging->metal);
  free(staging);
}

static GeozlGpuBatch *metal_batch_create(GeozlGpuProgram *program,
                                         const GeozlGpuBuffer buffers[],
                                         size_t bufferCount, char *errorBuffer,
                                         size_t errorCapacity) {
  @autoreleasepool {
    MetalBatch *batch = calloc(1, sizeof(*batch));
    if (batch == NULL) {
      geozl_gpu_error(errorBuffer, errorCapacity, "out of memory");
      return NULL;
    }
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
        batch->offsets[i] = batch->offsets[buffer->alias];
      } else if ((buffer->flags & GEOZL_GPU_EXTERNAL) != 0) {
        metalBuffer = metal_memory(buffer->memory).buffer;
        batch->offsets[i] = buffer->memoryOffset;
      } else if (buffer->staging != NULL) {
        metalBuffer = staging_buffer(buffer->staging);
        batch->offsets[i] = (size_t)((uint8_t *)buffer->host -
                                     (uint8_t *)buffer->staging->data);
      } else if (buffer->host == NULL) {
        const size_t length = blit_size(buffer->bytes);
        metalBuffer = length == 0 ? nil : acquire_private(metalProgram, length);
        batch->spare[i] = metalBuffer != nil;
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
        for (size_t b = 0; b < i; ++b) {
          if (batch->spare[b])
            release_private(metalProgram, metalBuffers[b]);
        }
        free(batch);
        return NULL;
      }
      [metalBuffers addObject:metalBuffer];
    }

    GeozlMetalBatch *metal = [GeozlMetalBatch new];
    metal.program = metalProgram;
    metal.buffers = metalBuffers;
    geozl_gpu_batch_init(&batch->base, program, buffers, bufferCount);
    batch->metal = (__bridge_retained void *)metal;
    return &batch->base;
  }
}

static void metal_batch_destroy(GeozlGpuBatch *base) {
  MetalBatch *batch = (MetalBatch *)base;
  @autoreleasepool {
    GeozlMetalBatch *metal = metal_batch(base);
    for (size_t i = 0; i < base->bufferCount; ++i) {
      if (batch->spare[i])
        release_private(metal.program, metal.buffers[i]);
    }
  }
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

static void upload_inputs(GeozlGpuBatch *batch) {
  GeozlMetalBatch *metal = metal_batch(batch);
  const size_t *offsets = ((const MetalBatch *)batch)->offsets;
  for (size_t i = 0; i < batch->bufferCount; ++i) {
    const GeozlGpuBuffer *buffer = &batch->buffers[i];
    if (buffer->alias >= 0 || (buffer->flags & GEOZL_GPU_COPY_IN) == 0)
      continue;
    uint8_t *contents = (uint8_t *)metal.buffers[i].contents + offsets[i];
    if (contents != buffer->host)
      memcpy(contents, buffer->host, buffer->bytes);
  }
}

static int metal_batch_run(GeozlGpuBatch *batch,
                           const GeozlGpuDispatch dispatches[],
                           size_t dispatchCount, GeozlGpuRunTimes *times,
                           char *errorBuffer, size_t errorCapacity) {
  @autoreleasepool {
    GeozlMetalBatch *metal = metal_batch(batch);
    const size_t *offsets = ((const MetalBatch *)batch)->offsets;
    const double uploadStart = seconds_now();
    upload_inputs(batch);
    times->upload = seconds_now() - uploadStart;

    // One command buffer per dispatch gives each kernel its own GPU time.
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
        [encoder setBuffer:metal.buffers[b] offset:offsets[b] atIndex:b];
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

// Programs have queues of their own, so a batch finishes before enqueue
// returns and the next batch, on any queue, sees its output.
static int metal_batch_enqueue(GeozlGpuBatch *batch,
                               const GeozlGpuDispatch dispatches[],
                               size_t dispatchCount, void *stream,
                               char *errorBuffer, size_t errorCapacity) {
  (void)stream;
  @autoreleasepool {
    GeozlMetalBatch *metal = metal_batch(batch);
    const size_t *offsets = ((const MetalBatch *)batch)->offsets;
    upload_inputs(batch);
    id<MTLCommandBuffer> command = [metal.program.queue commandBuffer];
    if (command == nil)
      return geozl_gpu_error(errorBuffer, errorCapacity,
                             "Metal command allocation failed");
    for (size_t i = 0; i < dispatchCount; ++i) {
      const GeozlGpuDispatch *dispatch = &dispatches[i];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      if (encoder == nil)
        return geozl_gpu_error(errorBuffer, errorCapacity,
                               "Metal command allocation failed");
      [encoder
          setComputePipelineState:metal.program.pipelines[dispatch->kernel]];
      for (size_t b = 0; b < batch->bufferCount; ++b)
        [encoder setBuffer:metal.buffers[b] offset:offsets[b] atIndex:b];
      [encoder dispatchThreadgroups:MTLSizeMake(dispatch->groupsX,
                                                dispatch->groupsY, 1)
              threadsPerThreadgroup:MTLSizeMake(dispatch->threads, 1, 1)];
      [encoder endEncoding];
    }
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted)
      return geozl_gpu_error(errorBuffer, errorCapacity,
                             "Metal kernels failed: %s",
                             metal_message(command.error));
    return 0;
  }
}

static int metal_batch_wait(GeozlGpuBatch *batch, char *errorBuffer,
                            size_t errorCapacity) {
  (void)batch;
  (void)errorBuffer;
  (void)errorCapacity;
  return 0;
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
    ((MetalBatch *)batch)->spare[bufferIndex] = 0;
    return &memory->base;
  }
}

static GeozlGpuMemory *new_memory(GeozlMetalProgram *program,
                                  id<MTLBuffer> buffer, char *errorBuffer,
                                  size_t errorCapacity) {
  GeozlMetalMemory *metalMemory = [GeozlMetalMemory new];
  metalMemory.device = program.device;
  metalMemory.queue = program.queue;
  metalMemory.buffer = buffer;
  MetalMemory *memory = calloc(1, sizeof(*memory));
  if (memory == NULL) {
    geozl_gpu_error(errorBuffer, errorCapacity, "out of memory");
    return NULL;
  }
  memory->metal = (__bridge_retained void *)metalMemory;
  return &memory->base;
}

static GeozlGpuMemory *metal_memory_create(GeozlGpuProgram *program,
                                           size_t bytes, char *errorBuffer,
                                           size_t errorCapacity) {
  @autoreleasepool {
    const size_t length = blit_size(bytes);
    id<MTLBuffer> buffer =
        length == 0 ? nil
                    : [metal_program(program).device
                          newBufferWithLength:length
                                      options:MTLResourceStorageModePrivate];
    if (buffer == nil) {
      geozl_gpu_error(errorBuffer, errorCapacity,
                      "Metal memory allocation failed");
      return NULL;
    }
    return new_memory(metal_program(program), buffer, errorBuffer,
                      errorCapacity);
  }
}

static GeozlGpuMemory *metal_memory_wrap(GeozlGpuProgram *program, void *data,
                                         size_t bytes, char *errorBuffer,
                                         size_t errorCapacity) {
  @autoreleasepool {
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)data;
    if (buffer.length < bytes ||
        buffer.device != metal_program(program).device) {
      geozl_gpu_error(errorBuffer, errorCapacity,
                      "Metal buffer does not fit the wrapped memory");
      return NULL;
    }
    return new_memory(metal_program(program), buffer, errorBuffer,
                      errorCapacity);
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
    .staging_create = metal_staging_create,
    .staging_destroy = metal_staging_destroy,
    .batch_create = metal_batch_create,
    .batch_destroy = metal_batch_destroy,
    .dispatches_fit = metal_dispatches_fit,
    .batch_run = metal_batch_run,
    .batch_enqueue = metal_batch_enqueue,
    .batch_wait = metal_batch_wait,
    .batch_download = metal_batch_download,
    .batch_take = metal_batch_take,
    .memory_create = metal_memory_create,
    .memory_wrap = metal_memory_wrap,
    .memory_destroy = metal_memory_destroy,
    .memory_data = metal_memory_data,
    .memory_download = metal_memory_download,
};
