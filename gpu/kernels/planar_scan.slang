// Inverse planar reconstruction. Wide planes use separate row and column
// passes; scan_planes handles widths up to SEGMENT in one pass.

uint unzigzag(uint v) {
  return ((v >> 1) ^ (0u - (v & 1u))) & low_mask(8u * ELT_BYTES);
}

bool aligned_run(uint elt) {
  return ((elt * ELT_BYTES) & (CPT * ELT_BYTES - 1u)) == 0;
}

void store_run(uint elt, uint v[CPT]) {
#if ELT_BYTES == 1
  outputBytes.Store2(
      elt,
      uint2((v[0] & 0xFFu) | ((v[1] & 0xFFu) << 8) |
                ((v[2] & 0xFFu) << 16) | (v[3] << 24),
            (v[4] & 0xFFu) | ((v[5] & 0xFFu) << 8) |
                ((v[6] & 0xFFu) << 16) | (v[7] << 24)));
#else
  outputBytes.Store4(
      elt * 2u,
      uint4((v[0] & 0xFFFFu) | (v[1] << 16),
            (v[2] & 0xFFFFu) | (v[3] << 16),
            (v[4] & 0xFFFFu) | (v[5] << 16),
            (v[6] & 0xFFFFu) | (v[7] << 16)));
#endif
}

void load_run(uint elt, out uint v[CPT]) {
#if ELT_BYTES == 1
  uint2 w = outputBytes.Load2(elt);
  [ForceUnroll]
  for (uint c = 0; c < 4; ++c) {
    v[c] = (w.x >> (c * 8u)) & 0xFFu;
    v[4 + c] = (w.y >> (c * 8u)) & 0xFFu;
  }
#else
  uint4 w = outputBytes.Load4(elt * 2u);
  v[0] = w.x & 0xFFFFu;
  v[1] = w.x >> 16;
  v[2] = w.y & 0xFFFFu;
  v[3] = w.y >> 16;
  v[4] = w.z & 0xFFFFu;
  v[5] = w.z >> 16;
  v[6] = w.w & 0xFFFFu;
  v[7] = w.w >> 16;
#endif
}

void load_row(uint o0, uint x0, uint width, out uint r[CPT]) {
  if (x0 + CPT <= width && aligned_run(o0 + x0))
    load_run(o0 + x0, r);
  else
    [ForceUnroll]
    for (uint c = 0; c < CPT; ++c)
      r[c] = (x0 + c < width) ? uint(output[o0 + x0 + c]) : 0u;
}

void store_row(uint o0, uint x0, uint width, uint r[CPT]) {
  if (x0 + CPT <= width && aligned_run(o0 + x0))
    store_run(o0 + x0, r);
  else
    [ForceUnroll]
    for (uint c = 0; c < CPT; ++c)
      if (x0 + c < width)
        output[o0 + x0 + c] = Elt(r[c]);
}

[shader("compute")]
[numthreads(WAVES_PER_GROUP * WAVE, 1, 1)]
void scan_planes(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
  uint t = thread.x % WAVE;
  plane_task task = planes[group.x * WAVES_PER_GROUP + thread.x / WAVE];
  if (task.width > SEGMENT || task.rows == 0)
    return;
  uint x0 = t * CPT;

  uint column[CPT];
  uint next[CPT];
  [ForceUnroll]
  for (uint c = 0; c < CPT; ++c)
    column[c] = 0;
  load_row(task.outBase, x0, task.width, next);
  for (uint y = 0; y < task.rows; ++y) {
    uint o0 = task.outBase + y * task.width;
    uint r[CPT];
    [ForceUnroll]
    for (uint c = 0; c < CPT; ++c)
      r[c] = next[c];
    if (y + 1 < task.rows)
      load_row(o0 + task.width, x0, task.width, next);
    uint local = 0;
    [ForceUnroll]
    for (uint c = 0; c < CPT; ++c) {
      local += r[c];
      r[c] = local;
    }
    uint before = WavePrefixSum(local);
    [ForceUnroll]
    for (uint c = 0; c < CPT; ++c) {
      column[c] += r[c] + before;
      r[c] = column[c];
    }
    store_row(o0, x0, task.width, r);
  }
}

// Group (p, y) holds rows y * WAVES_PER_GROUP.. of plane p, one wave per row.
[shader("compute")]
[numthreads(WAVES_PER_GROUP * WAVE, 1, 1)]
void scan_rows(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
  uint t = thread.x % WAVE;
  plane_task task = planes[group.x];
  uint y = group.y * WAVES_PER_GROUP + thread.x / WAVE;
  if (task.width <= SEGMENT || y >= task.rows)
    return;
  uint o0 = task.outBase + y * task.width;
  uint rowCarry = 0;
  for (uint xs = t * CPT; xs < task.width; xs += SEGMENT) {
    uint r[CPT];
    load_row(o0, xs, task.width, r);
    uint local = 0;
    [ForceUnroll]
    for (uint c = 0; c < CPT; ++c) {
      local += r[c];
      r[c] = local;
    }
    uint before = WavePrefixSum(local) + rowCarry;
    rowCarry += WaveActiveSum(local);
    [ForceUnroll]
    for (uint c = 0; c < CPT; ++c)
      r[c] += before;
    store_row(o0, xs, task.width, r);
  }
}

// Group (p, g) covers columns g * COLUMNS_PER_GROUP.. of plane p, one per thread.
static const uint COLUMNS_PER_GROUP = WAVES_PER_GROUP * WAVE;

[shader("compute")]
[numthreads(COLUMNS_PER_GROUP, 1, 1)]
void scan_columns(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
  plane_task task = planes[group.x];
  uint x = group.y * COLUMNS_PER_GROUP + thread.x;
  if (task.width <= SEGMENT || x >= task.width)
    return;
  uint carry = 0;
  for (uint y = 0; y < task.rows; ++y) {
    uint o = task.outBase + y * task.width + x;
    carry += uint(output[o]);
    output[o] = Elt(carry);
  }
}
