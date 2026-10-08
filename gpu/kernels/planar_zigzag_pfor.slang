// GPU decoder for planar_zigzag_pfor (0x72D710).

#ifndef ELT_BYTES
#define ELT_BYTES 2
#endif

#if ELT_BYTES == 1
typedef uint8_t Elt;
#elif ELT_BYTES == 2
typedef uint16_t Elt;
#else
#error "ELT_BYTES must be 1 or 2"
#endif

#define PFOR_BYTES ELT_BYTES
#include "payload.slang"
#include "pfor_block.slang"

struct block_task {
  uint payOff;
  uint outBase;
  uint count;
};

struct plane_task {
  uint outBase;
  uint width;
  uint rows;
};

StructuredBuffer<block_task> blocks;
StructuredBuffer<plane_task> planes;
RWStructuredBuffer<Elt> output;
RWByteAddressBuffer outputBytes;

#include "planar_scan.slang"

[shader("compute")]
[numthreads(WAVES_PER_GROUP * WAVE, 1, 1)]
void unpack_blocks(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
  uint t = thread.x % WAVE;
  block_task task = blocks[group.x * WAVES_PER_GROUP + thread.x / WAVE];
  uint v[CPT];
  pfor_lane(task.payOff, t, v);
  [ForceUnroll]
  for (uint c = 0; c < CPT; ++c)
    v[c] = unzigzag(v[c]);
  uint i0 = t * CPT;
  uint elt = task.outBase + i0;
  if (i0 + CPT <= task.count && aligned_run(elt))
    store_run(elt, v);
  else
    [ForceUnroll]
    for (uint c = 0; c < CPT; ++c)
      if (i0 + c < task.count)
        output[elt + c] = Elt(v[c]);
}
