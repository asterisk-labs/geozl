// GPU decoder program for PivCo-backed GeoZL graphs.

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

#define PFOR_BYTES 1
#include "../common/payload.slang"
#include "../stages/pfor.slang"
#include "../stages/pivco.slang"

struct lane_block {
  uint payOff;
  uint outBase;
};

struct join_task {
  uint laneBase;
  uint laneStride;
  uint count;
  uint outBase;
  uint start;
  uint transform;
};

struct plane_task {
  uint outBase;
  uint width;
  uint rows;
};

StructuredBuffer<lane_block> laneBlocks;
StructuredBuffer<join_task> joins;
StructuredBuffer<plane_task> planes;
RWStructuredBuffer<Elt> output;
RWByteAddressBuffer outputBytes;
RWStructuredBuffer<uint8_t> scratch;

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
void unpack_lane_blocks(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
  uint t = thread.x % WAVE;
  lane_block blk = laneBlocks[group.x * WAVES_PER_GROUP + thread.x / WAVE];
  uint v[CPT];
  pfor_lane(blk.payOff, t, v);
  lanes.Store2(blk.outBase + t * CPT,
               uint2(v[0] | (v[1] << 8) | (v[2] << 16) | (v[3] << 24),
                     v[4] | (v[5] << 8) | (v[6] << 16) | (v[7] << 24)));
}

// Each group joins one 1024-sample chunk from one frame.
[shader("compute")]
[numthreads(WAVES_PER_GROUP * WAVE, 1, 1)]
void join_lanes(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
  join_task f = joins[group.x];
  uint i0 = f.start + thread.x * CPT;
  if (i0 >= f.count)
    return;
  uint v[CPT];
  [ForceUnroll]
  for (uint c = 0; c < CPT; ++c)
    v[c] = 0;
  [ForceUnroll]
  for (uint k = 0; k < ELT_BYTES; ++k) {
    uint2 w = lanes.Load2(f.laneBase + k * f.laneStride + i0);
    [ForceUnroll]
    for (uint c = 0; c < 4; ++c) {
      v[c] |= ((w.x >> (c * 8u)) & 0xFFu) << (8u * k);
      v[4 + c] |= ((w.y >> (c * 8u)) & 0xFFu) << (8u * k);
    }
  }
  if (f.transform != 0)
    [ForceUnroll]
    for (uint c = 0; c < CPT; ++c)
      v[c] = unzigzag(v[c]);
  uint elt = f.outBase + i0;
  if (i0 + CPT <= f.count && aligned_run(elt))
    store_run(elt, v);
  else
    [ForceUnroll]
    for (uint c = 0; c < CPT; ++c)
      if (i0 + c < f.count)
        output[elt + c] = Elt(v[c]);
}

[shader("compute")]
[numthreads(WAVES_PER_GROUP * WAVE, 1, 1)]
void copy_raw(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
  raw_copy task = rawCopies[group.x];
  for (uint i = thread.x; i < task.count; i += WAVES_PER_GROUP * WAVE)
    output[task.destination + i] = rawValues[task.source + i];
}
