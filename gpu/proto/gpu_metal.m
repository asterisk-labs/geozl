// Metal backend of gpu.h. Apple GPUs share memory with the CPU: page-aligned
// host buffers are wrapped in place, so neither input nor output is copied.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "gpu.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

const char *gpu_backend_name = "metal";

double now_seconds(void);

static int validate_run(const GpuBuffer *buffers, int nBuffers,
                        const GpuDispatch *dispatches, int nDispatches,
                        int reps, const GpuTimes *times) {
  if (buffers == NULL || dispatches == NULL || times == NULL || reps <= 0 ||
      nBuffers <= 0 || nBuffers > GPU_MAX_BUFFERS || nDispatches <= 0 ||
      nDispatches > GPU_MAX_DISPATCHES) {
    fprintf(stderr, "invalid GPU run dimensions\n");
    return 1;
  }
  for (int i = 0; i < nBuffers; ++i) {
    const GpuBuffer *buffer = &buffers[i];
    if (buffer->bytes == 0 || buffer->elem == 0 ||
        buffer->bytes % buffer->elem != 0 || buffer->aliasOf >= i ||
        buffer->aliasOf < -1) {
      fprintf(stderr, "invalid GPU buffer %d\n", i);
      return 1;
    }
  }
  for (int i = 0; i < nDispatches; ++i) {
    const GpuDispatch *dispatch = &dispatches[i];
    if (dispatch->entry == NULL || dispatch->groupsX == 0 ||
        dispatch->groupsY == 0 || dispatch->threads == 0) {
      fprintf(stderr, "invalid GPU dispatch %d\n", i);
      return 1;
    }
  }
  return 0;
}

int gpu_run(const char *module, GpuBuffer *buffers, int nBuffers,
            const GpuDispatch *dispatches, int nDispatches, int reps,
            GpuTimes *times) {
  if (module == NULL || validate_run(buffers, nBuffers, dispatches, nDispatches,
                                     reps, times) != 0)
    return 1;
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      fprintf(stderr, "Metal device unavailable\n");
      return 1;
    }
    snprintf(times->device, sizeof(times->device), "%s",
             device.name.UTF8String);
    NSError *error = nil;
    NSString *source = [NSString stringWithContentsOfFile:@(module)
                                                 encoding:NSUTF8StringEncoding
                                                    error:&error];
    if (!source) {
      fprintf(stderr, "cannot read %s\n", module);
      return 1;
    }
    id<MTLLibrary> library = [device newLibraryWithSource:source
                                                  options:nil
                                                    error:&error];
    if (!library) {
      const char *message =
          error ? error.localizedDescription.UTF8String : NULL;
      fprintf(stderr, "cannot compile %s: %s\n", module,
              message ? message : "Metal returned no diagnostic");
      return 1;
    }
    id<MTLComputePipelineState> pipelines[GPU_MAX_DISPATCHES];
    for (int d = 0; d < nDispatches; ++d) {
      id<MTLFunction> fn = [library newFunctionWithName:@(dispatches[d].entry)];
      pipelines[d] = fn ? [device newComputePipelineStateWithFunction:fn
                                                                error:&error]
                        : nil;
      if (!pipelines[d]) {
        fprintf(stderr, "no pipeline for %s\n", dispatches[d].entry);
        return 1;
      }
      // Kernels scan with 32-lane waves.
      if (pipelines[d].threadExecutionWidth != 32) {
        fprintf(stderr, "needs a 32-wide SIMD group\n");
        return 1;
      }
      if (dispatches[d].threads > pipelines[d].maxTotalThreadsPerThreadgroup) {
        fprintf(stderr, "%s requests %u threads, device allows %lu\n",
                dispatches[d].entry, dispatches[d].threads,
                (unsigned long)pipelines[d].maxTotalThreadsPerThreadgroup);
        return 1;
      }
    }

    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
    id<MTLBuffer> bufs[GPU_MAX_BUFFERS];
    for (int i = 0; i < nBuffers; ++i) {
      GpuBuffer *b = &buffers[i];
      if (b->aliasOf >= 0) {
        bufs[i] = bufs[b->aliasOf];
        continue;
      }
      if (!b->host)
        bufs[i] = [device newBufferWithLength:b->bytes
                                      options:MTLResourceStorageModePrivate];
      else if (((uintptr_t)b->host % page) == 0 && b->bytes % page == 0)
        bufs[i] = [device newBufferWithBytesNoCopy:b->host
                                            length:b->bytes
                                           options:MTLResourceStorageModeShared
                                       deallocator:nil];
      else
        bufs[i] = [device newBufferWithBytes:b->host
                                      length:b->bytes
                                     options:MTLResourceStorageModeShared];
      if (!bufs[i]) {
        fprintf(stderr, "buffer %d allocation failed\n", i);
        return 1;
      }
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) {
      fprintf(stderr, "Metal command queue allocation failed\n");
      return 1;
    }

    times->kernel = times->endToEnd = 1e30;
    times->copy = 0;
    for (int d = 0; d < nDispatches; ++d)
      times->pass[d] = 1e30;
    for (int r = 0; r < reps; ++r) {
      // One command buffer per dispatch times each; the queue runs them in
      // order.
      const double t0 = now_seconds();
      id<MTLCommandBuffer> cmds[GPU_MAX_DISPATCHES];
      for (int d = 0; d < nDispatches; ++d) {
        cmds[d] = [queue commandBuffer];
        if (!cmds[d]) {
          fprintf(stderr, "Metal command buffer allocation failed\n");
          return 1;
        }
        id<MTLComputeCommandEncoder> enc = [cmds[d] computeCommandEncoder];
        if (!enc) {
          fprintf(stderr, "Metal command encoder allocation failed\n");
          return 1;
        }
        [enc setComputePipelineState:pipelines[d]];
        for (int i = 0; i < nBuffers; ++i)
          [enc setBuffer:bufs[i] offset:0 atIndex:(NSUInteger)i];
        [enc dispatchThreadgroups:MTLSizeMake(dispatches[d].groupsX,
                                              dispatches[d].groupsY, 1)
            threadsPerThreadgroup:MTLSizeMake(dispatches[d].threads, 1, 1)];
        [enc endEncoding];
        [cmds[d] commit];
      }
      [cmds[nDispatches - 1] waitUntilCompleted];
      const double wall = now_seconds() - t0;
      double total = 0;
      for (int d = 0; d < nDispatches; ++d) {
        [cmds[d] waitUntilCompleted];
        if (cmds[d].status != MTLCommandBufferStatusCompleted) {
          fprintf(stderr, "%s failed\n", dispatches[d].entry);
          return 1;
        }
        const double t = cmds[d].GPUEndTime - cmds[d].GPUStartTime;
        total += t;
        if (t < times->pass[d])
          times->pass[d] = t;
      }
      if (total < times->kernel)
        times->kernel = total;
      if (wall < times->endToEnd)
        times->endToEnd = wall;
    }
    for (int i = 0; i < nBuffers; ++i)
      if (buffers[i].output && buffers[i].host && buffers[i].aliasOf < 0 &&
          bufs[i].contents != buffers[i].host)
        memcpy(buffers[i].host, bufs[i].contents, buffers[i].bytes);
    return 0;
  }
}
