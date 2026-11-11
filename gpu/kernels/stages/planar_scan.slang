// In-place inverse planar scan for the lanes codec.

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
  uint segments = (task.width + SEGMENT - 1u) / SEGMENT;
  for (uint segment = 0; segment < segments; ++segment) {
    uint xs = segment * SEGMENT + t * CPT;
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
