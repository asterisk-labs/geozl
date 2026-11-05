#include "pfor_plan.h"

#include "pfor/pfor_check.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(sizeof(GpuPforBlock) == 16, "GpuPforBlock must match Slang");

static unsigned popcount64(uint64_t value) {
  value -= (value >> 1) & 0x5555555555555555u;
  value = (value & 0x3333333333333333u) + ((value >> 2) & 0x3333333333333333u);
  value = (value + (value >> 4)) & 0x0F0F0F0F0F0F0F0Fu;
  return (unsigned)((value * 0x0101010101010101u) >> 56);
}

int pfor_gpu_block_info(const uint8_t *block, size_t available,
                        size_t elementBytes, PforGpuBlockInfo *info) {
  if (block == NULL || info == NULL || !geozl_pfor_width_ok(elementBytes) ||
      available < 2)
    return 1;

  const unsigned bits = block[0];
  const unsigned exceptions = block[1];
  if (bits > elementBytes * 8)
    return 1;

  size_t exceptionBytes = 0;
  if (exceptions != 0) {
    if (available < 3)
      return 1;
    const unsigned descriptor = block[2];
    const unsigned exceptionBits = descriptor & 0x7Fu;
    const unsigned mode = descriptor >> 7;
    if (exceptionBits == 0 || bits + exceptionBits > elementBytes * 8)
      return 1;
    exceptionBytes = geozl_pfor_exc_bytes(exceptions, exceptionBits, mode);
  }

  const size_t bodyBytes = geozl_pfor_body_bytes(bits);
  if (exceptionBytes > SIZE_MAX - 2 ||
      bodyBytes > SIZE_MAX - 2 - exceptionBytes)
    return 1;
  const size_t bytes = 2 + exceptionBytes + bodyBytes;
  if (bytes > available)
    return 1;
  if (exceptions != 0) {
    const uint8_t *positions = block + 3;
    const unsigned mode = block[2] >> 7;
    if (mode == GEOZL_PFOR_MODE_BITMAP) {
      unsigned count = 0;
      for (size_t i = 0; i < 32; i += 8) {
        uint64_t word;
        memcpy(&word, positions + i, sizeof(word));
        count += popcount64(word);
      }
      if (count != exceptions)
        return 1;
    } else {
      for (unsigned i = 1; i < exceptions; ++i) {
        if (positions[i - 1] >= positions[i])
          return 1;
      }
    }
  }
  *info = (PforGpuBlockInfo){bytes, exceptions};
  return 0;
}

void pfor_gpu_plan_free(PforGpuPlan *plan) {
  if (plan == NULL)
    return;
  if (!plan->borrowed)
    free(plan->blocks);
  *plan = (PforGpuPlan){0};
}

int pfor_gpu_plan_frame(GpuPforBlock blocks[], const PforGpuFrame *frame,
                        const uint8_t *payload, size_t elementBytes,
                        PforGpuFrameTally *tally) {
  *tally = (PforGpuFrameTally){.errorBlock = SIZE_MAX};
  size_t at = 0;
  size_t output = frame->shape.outOff;
  for (size_t block = 0; block * GEOZL_PFOR_BLOCK < frame->shape.nbElts;
       ++block) {
    PforGpuBlockInfo info;
    if (pfor_gpu_block_info(payload + frame->payloadOffset + at,
                            frame->payloadSize - at, elementBytes,
                            &info) != 0) {
      tally->error = "invalid PFOR block";
      tally->errorBlock = block;
      return 1;
    }
    const size_t left = frame->shape.nbElts - block * GEOZL_PFOR_BLOCK;
    const size_t count = left < GEOZL_PFOR_BLOCK ? left : GEOZL_PFOR_BLOCK;
    if (frame->payloadOffset + at > UINT32_MAX || output > UINT32_MAX) {
      tally->error = "PFOR task exceeds 32-bit kernel offsets";
      tally->errorBlock = block;
      return 1;
    }
    blocks[block] = (GpuPforBlock){(uint32_t)(frame->payloadOffset + at),
                                   (uint32_t)output, (uint32_t)count,
                                   (uint32_t)frame->transform};
    if (info.exceptions != 0) {
      ++tally->blocksWithExceptions;
      tally->exceptionCount += info.exceptions;
    }
    at += info.bytes;
    output += count;
  }
  if (at != frame->payloadSize) {
    tally->error = "PFOR frame has trailing bytes";
    return 1;
  }
  return 0;
}

// Four waves share a group and a padding task still reads its block. It repeats
// a checked one, since the payload may start with a frame the host decodes.
static void pad_blocks(GpuPforBlock blocks[], size_t realBlocks,
                       size_t paddedBlocks) {
  for (size_t i = realBlocks; i < paddedBlocks; ++i)
    blocks[i] = (GpuPforBlock){blocks[0].payloadOffset, 0, 0, 0};
}

void pfor_gpu_plan_wrap(PforGpuPlan *plan, GpuPforBlock blocks[],
                        size_t realBlocks) {
  const size_t paddedBlocks = (realBlocks + 3) / 4 * 4;
  pad_blocks(blocks, realBlocks, paddedBlocks);
  *plan = (PforGpuPlan){.blocks = blocks,
                        .blockCount = paddedBlocks,
                        .realBlockCount = realBlocks,
                        .borrowed = 1};
}

static int fail(PforGpuPlan *plan, const char *message, size_t frame,
                size_t block) {
  pfor_gpu_plan_free(plan);
  plan->error = message;
  plan->errorFrame = frame;
  plan->errorBlock = block;
  return 1;
}

int pfor_gpu_plan_build(PforGpuPlan *plan, const PforGpuFrame *frames,
                        size_t frameCount, const uint8_t *payload,
                        size_t payloadSize, size_t elementBytes) {
  if (plan == NULL)
    return 1;
  *plan = (PforGpuPlan){0};
  if (frames == NULL || payload == NULL || frameCount == 0 ||
      (elementBytes != 1 && elementBytes != 2)) {
    plan->error = "invalid PFOR GPU plan input";
    plan->errorFrame = SIZE_MAX;
    plan->errorBlock = SIZE_MAX;
    return 1;
  }

  size_t realBlocks = 0;
  for (size_t i = 0; i < frameCount; ++i) {
    const PforGpuFrame *frame = &frames[i];
    if (frame->shape.nbElts == 0 || frame->payloadSize == 0 ||
        (frame->transform != PFOR_GPU_ID &&
         frame->transform != PFOR_GPU_PLANAR_ZIGZAG) ||
        frame->payloadOffset > payloadSize ||
        frame->payloadSize > payloadSize - frame->payloadOffset ||
        frame->shape.nbElts > SIZE_MAX - (GEOZL_PFOR_BLOCK - 1))
      return fail(plan, "invalid PFOR frame", i, SIZE_MAX);
    const size_t blocks =
        (frame->shape.nbElts + GEOZL_PFOR_BLOCK - 1) / GEOZL_PFOR_BLOCK;
    if (blocks > SIZE_MAX - realBlocks)
      return fail(plan, "PFOR block count overflows", i, SIZE_MAX);
    realBlocks += blocks;
  }
  if (realBlocks > SIZE_MAX - 3)
    return fail(plan, "PFOR block count overflows", SIZE_MAX, SIZE_MAX);
  const size_t paddedBlocks = (realBlocks + 3) / 4 * 4;
  if (paddedBlocks > SIZE_MAX / sizeof(*plan->blocks))
    return fail(plan, "PFOR task bytes overflow", SIZE_MAX, SIZE_MAX);
  plan->blocks = calloc(paddedBlocks, sizeof(*plan->blocks));
  if (plan->blocks == NULL)
    return fail(plan, "out of memory", SIZE_MAX, SIZE_MAX);

  size_t task = 0;
  for (size_t i = 0; i < frameCount; ++i) {
    const PforGpuFrame *frame = &frames[i];
    PforGpuFrameTally tally;
    if (pfor_gpu_plan_frame(plan->blocks + task, frame, payload, elementBytes,
                            &tally) != 0)
      return fail(plan, tally.error, i, tally.errorBlock);
    task += (frame->shape.nbElts + GEOZL_PFOR_BLOCK - 1) / GEOZL_PFOR_BLOCK;
    plan->blocksWithExceptions += tally.blocksWithExceptions;
    if (tally.exceptionCount > SIZE_MAX - plan->exceptionCount)
      return fail(plan, "PFOR exception count overflows", i, SIZE_MAX);
    plan->exceptionCount += tally.exceptionCount;
  }

  pad_blocks(plan->blocks, realBlocks, paddedBlocks);
  plan->realBlockCount = realBlocks;
  plan->blockCount = paddedBlocks;
  return 0;
}
