// Zigzag and the row loads and stores shared by the planar decoders.

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
