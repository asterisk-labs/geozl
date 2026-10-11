// GPU decoder program for PFOR-backed GeoZL graphs.

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
#include "../common/payload.slang"
#include "../stages/pfor.slang"

struct block_task {
  uint payOff;
  uint outBase;
  uint count;
  uint transform;
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

struct raw_copy {
  uint source;
  uint destination;
  uint count;
};

StructuredBuffer<Elt> rawValues;
StructuredBuffer<raw_copy> rawCopies;

#include "../stages/planar_zigzag.slang"

[shader("compute")]
[numthreads(WAVES_PER_GROUP * WAVE, 1, 1)]
void unpack_blocks(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
  uint t = thread.x % WAVE;
  block_task task = blocks[group.x * WAVES_PER_GROUP + thread.x / WAVE];
  uint v[CPT];
  pfor_lane(task.payOff, t, v);
  if (task.transform != 0)
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

[shader("compute")]
[numthreads(WAVES_PER_GROUP * WAVE, 1, 1)]
void copy_raw(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
  raw_copy task = rawCopies[group.x];
  for (uint i = thread.x; i < task.count; i += WAVES_PER_GROUP * WAVE)
    output[task.destination + i] = rawValues[task.source + i];
}
