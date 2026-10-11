// Shared payload loads. Include this first to keep payload at binding 0.

static const uint WAVE = 32;
static const uint WAVES_PER_GROUP = 4;
static const uint CPT = 8;
static const uint SEGMENT = WAVE * CPT;

ByteAddressBuffer payload;

uint low_mask(uint b) {
  return b >= 32u ? 0xFFFFFFFFu : ((1u << b) - 1u);
}

uint load8(uint a) {
  return (payload.Load(a & ~3u) >> ((a & 3u) * 8u)) & 0xFFu;
}

uint load32(uint a) {
  uint shift = (a & 3u) * 8u;
  uint lo = payload.Load(a & ~3u);
  if (shift == 0)
    return lo;
  uint hi = payload.Load((a & ~3u) + 4u);
  return (lo >> shift) | (hi << (32u - shift));
}

void load128(uint a, out uint w[4]) {
  uint base = a & ~3u;
  uint shift = (a & 3u) * 8u;
  uint4 lo = payload.Load4(base);
  uint hi = payload.Load(base + 16u);
  if (shift == 0) {
    w[0] = lo.x;
    w[1] = lo.y;
    w[2] = lo.z;
    w[3] = lo.w;
    return;
  }
  w[0] = (lo.x >> shift) | (lo.y << (32u - shift));
  w[1] = (lo.y >> shift) | (lo.z << (32u - shift));
  w[2] = (lo.z >> shift) | (lo.w << (32u - shift));
  w[3] = (lo.w >> shift) | (hi << (32u - shift));
}
