// GPU path for the banded layout in planar_zigzag_pfor/spec.md.
// Include after planar_zigzag.slang.

struct band_frame {
  uint payOff;
  uint paySize;
  uint blockCount;
  uint firstBlock;
  uint frame; // the caller's index, for status
  uint pad0;
  uint pad1;
  uint pad2;
};

struct block_ref {
  uint hdr;
  uint rest;
};

struct block_chunk {
  uint frame;      // its band_frame
  uint first;      // its first block in the frame
  uint count;
  uint firstChunk; // the frame's first chunk
};

struct strip_task {
  uint firstBlock;
  uint blockCount;
  uint outBase; // the strip's first sample in the output
  uint width;   // the row width, which is the row stride
  uint height;  // 16, 8, 4, 2 or 1
  uint frame;
  uint columns; // the strip's own width
  uint pad;
};

StructuredBuffer<band_frame> bandFrames;
StructuredBuffer<strip_task> strips;
RWStructuredBuffer<block_ref> bandBlocks;
RWStructuredBuffer<uint> status;
StructuredBuffer<block_chunk> blockChunks;
RWStructuredBuffer<uint> chunkSizes;

static const uint ELT_BITS = 8u * ELT_BYTES;
static const uint BAD_BLOCK = 0xFFFFFFFFu;
static const uint STATUS_HEADER = 0u;
static const uint STATUS_POSITIONS = 1u;

// Explicit subgroup operations avoid Slang's per-read CUDA ballots.

uint wave_up(uint x, uint d) {
  __target_switch {
  case metal:
    __intrinsic_asm "simd_shuffle_up($0, ushort($1))";
  case cuda:
    __intrinsic_asm "__shfl_up_sync(0xffffffffu, $0, $1)";
  default:
    uint t = WaveGetLaneIndex();
    return WaveReadLaneAt(x, t >= d ? t - d : t);
  }
}

uint wave_xor(uint x, uint mask) {
  __target_switch {
  case metal:
    __intrinsic_asm "simd_shuffle_xor($0, ushort($1))";
  case cuda:
    __intrinsic_asm "__shfl_xor_sync(0xffffffffu, $0, $1)";
  default:
    return WaveReadLaneAt(x, WaveGetLaneIndex() ^ mask);
  }
}

uint wave_at(uint x, uint lane) {
  __target_switch {
  case metal:
    __intrinsic_asm "simd_shuffle($0, ushort($1))";
  case cuda:
    __intrinsic_asm "__shfl_sync(0xffffffffu, $0, $1)";
  default:
    return WaveReadLaneAt(x, lane);
  }
}

uint wave_broadcast(uint x, uint lane) {
  __target_switch {
  case metal:
    __intrinsic_asm "simd_broadcast($0, ushort($1))";
  case cuda:
    __intrinsic_asm "__shfl_sync(0xffffffffu, $0, $1)";
  default:
    return WaveReadLaneAt(x, lane);
  }
}

uint wave_inclusive(uint x) {
  __target_switch {
  case metal:
    __intrinsic_asm "simd_prefix_inclusive_sum($0)";
  default:
    return WavePrefixSum(x) + x;
  }
}

uint wave_or(uint x) {
  __target_switch {
  case metal:
    __intrinsic_asm "simd_or($0)";
  case cuda:
    [ForceUnroll]
    for (uint m = 1; m < WAVE; m *= 2)
      x |= wave_xor(x, m);
    return x;
  default:
    return WaveActiveBitOr(x);
  }
}

void report(uint check, uint frame) {
  InterlockedMin(status[check], frame);
}

uint sat_add(uint a, uint b) {
  uint sum = a + b;
  return sum < a ? 0xFFFFFFFFu : sum;
}

struct frame_rest {
  bool bad;   // the headers alone overrun the payload
  uint bytes; // what follows the headers
  uint cap;   // a sum this large has already left the frame
};

frame_rest rest_of(band_frame f) {
  frame_rest r;
  r.bad = f.blockCount > f.paySize / 3u;
  r.bytes = r.bad ? 0u : f.paySize - 3u * f.blockCount;
  r.cap = r.bytes + 1u;
  return r;
}

uint block_size(band_frame f, uint k, inout bool bad) {
  uint at = f.payOff + 3u * k;
  uint b = load8(at);
  uint nexc = load8(at + 1u);
  uint format = load8(at + 2u);
  uint size = 32u * b;
  if (b > ELT_BITS)
    bad = true;
  if (nexc != 0) {
    uint eb = format & 0x7Fu;
    if (eb == 0 || b + eb > ELT_BITS)
      bad = true;
    size += ((format >> 7) != 0 ? nexc : 32u) + ((nexc * eb + 7u) >> 3);
  } else if (format != 0) {
    bad = true;
  }
  return bad ? 0u : size;
}

void size_chunk(uint index, uint t) {
  block_chunk c = blockChunks[index];
  if (c.count == 0)
    return;
  band_frame f = bandFrames[c.frame];
  frame_rest r = rest_of(f);
  uint total = 0;
  if (!r.bad) {
    for (uint k0 = 0; k0 < c.count; k0 += WAVE) {
      uint k = k0 + t;
      bool bad = false;
      uint size = k < c.count ? block_size(f, c.first + k, bad) : 0u;
      total = min(total + WaveActiveSum(size), r.cap);
    }
  }
  if (t == 0)
    chunkSizes[index] = total;
}

void plan_chunk(uint index, uint t) {
  block_chunk c = blockChunks[index];
  if (c.count == 0)
    return;
  band_frame f = bandFrames[c.frame];
  frame_rest r = rest_of(f);
  uint restBase = f.payOff + 3u * f.blockCount;
  uint carry = 0;
  for (uint j = c.firstChunk + t; j < index; j += WAVE)
    carry = min(sat_add(carry, chunkSizes[j]), r.cap);
  [ForceUnroll]
  for (uint m = 1; m < WAVE; m *= 2)
    carry = min(sat_add(carry, wave_xor(carry, m)), r.cap);
  bool anyBad = false;
  for (uint k0 = 0; k0 < c.count; k0 += WAVE) {
    uint k = k0 + t;
    bool bad = r.bad;
    uint size = 0;
    if (k < c.count && !r.bad)
      size = block_size(f, c.first + k, bad);
    uint start = carry + WavePrefixSum(size);
    if (start + size > r.bytes)
      bad = true;
    if (k < c.count) {
      block_ref ref;
      ref.hdr = f.payOff + 3u * (c.first + k);
      ref.rest = bad ? BAD_BLOCK : restBase + start;
      bandBlocks[f.firstBlock + c.first + k] = ref;
    }
    anyBad = anyBad || WaveActiveAnyTrue(bad && k < c.count);
    carry = min(carry + WaveActiveSum(size), r.cap);
  }
  bool last = c.first + c.count == f.blockCount;
  if ((anyBad || r.bad || (last && carry != r.bytes)) && t == 0)
    report(STATUS_HEADER, f.frame);
}

void block_exceptions(uint rest, uint nexc, bool list, uint t,
                      out uint flags, out uint rank, inout bool bad) {
  flags = 0;
  rank = 0;
  uint q = t >> 2;
  uint shift = (t & 3u) * 8u;
  __target_switch {
  case metal:
  case cuda:
    uint mine = 0;
    uint total = 0;
    if (!list) {
      uint word = (t < 8u) ? load32(rest + t * 4u) : 0u;
      [ForceUnroll]
      for (uint w = 0; w < 8; ++w) {
        uint m = wave_broadcast(word, w);
        uint n = countbits(m);
        total += n;
        if (w < q)
          rank += n;
        if (w == q)
          mine = m;
      }
    } else {
      uint last = 0;
      for (uint chunk = 0; chunk * WAVE < nexc; ++chunk) {
        uint i = chunk * WAVE + t;
        uint p = (i < nexc) ? load8(rest + i) : 0xFFFFu;
        uint prev = wave_up(p, 1u);
        if (t == 0u)
          prev = last;
        if (i < nexc && i != 0u && prev >= p)
          bad = true;
        last = wave_broadcast(p, WAVE - 1u);
        [ForceUnroll]
        for (uint w = 0; w < 8; ++w) {
          uint m = wave_or((p >> 5) == w ? (1u << (p & 31u)) : 0u);
          uint n = countbits(m);
          total += n;
          if (w < q)
            rank += n;
          if (w == q)
            mine |= m;
        }
      }
    }
    if (total != nexc)
      bad = true;
    flags = (mine >> shift) & 0xFFu;
    rank += countbits(mine & low_mask(shift));
  default:
    if (!list) {
      uint w[8];
      uint a[4];
      load128(rest, a);
      w[0] = a[0];
      w[1] = a[1];
      w[2] = a[2];
      w[3] = a[3];
      load128(rest + 16u, a);
      w[4] = a[0];
      w[5] = a[1];
      w[6] = a[2];
      w[7] = a[3];
      uint total = 0;
      [ForceUnroll]
      for (uint k = 0; k < 8; ++k) {
        uint n = countbits(w[k]);
        total += n;
        if (k < q)
          rank += n;
        if (k == q) {
          flags = (w[k] >> shift) & 0xFFu;
          rank += countbits(w[k] & low_mask(shift));
        }
      }
      if (total != nexc)
        bad = true;
    } else {
      uint lo = t * CPT;
      uint prev = 0;
      for (uint i0 = 0; i0 < nexc; i0 += 4u) {
        uint word = load32(rest + i0);
        [ForceUnroll]
        for (uint k = 0; k < 4; ++k) {
          uint i = i0 + k;
          if (i < nexc) {
            uint p = (word >> (8u * k)) & 0xFFu;
            if (i != 0 && p <= prev)
              bad = true;
            prev = p;
            rank += p < lo ? 1u : 0u;
            uint d = p - lo;
            if (d < CPT)
              flags |= 1u << d;
          }
        }
      }
    }
  }
}

void block_lane(uint hdr, uint rest, uint t, out uint v[CPT], inout bool bad) {
  uint b = load8(hdr);
  uint nexc = load8(hdr + 1u);
  uint flags = 0;
  uint rank = 0;
  uint eb = 0;
  uint excStart = rest;
  uint body = rest;
  if (nexc != 0) {
    uint format = load8(hdr + 2u);
    eb = format & 0x7Fu;
    bool list = (format >> 7) != 0;
    block_exceptions(rest, nexc, list, t, flags, rank, bad);
    excStart = rest + (list ? nexc : 32u);
    body = excStart + ((nexc * eb + 7u) >> 3);
  }

  uint i0 = t * CPT;
  uint row[8];
  [ForceUnroll]
  for (uint c = 0; c < 8; ++c)
    row[c] = 0;
  uint off = 0;
  if (b != 0) {
    uint bitpos = (i0 / PFOR_LANES) * b;
    off = bitpos % PFOR_WORD_BITS;
    uint a = body + (bitpos / PFOR_WORD_BITS) * 16u +
             (i0 % PFOR_LANES) * PFOR_BYTES;
    uint w4[4];
    load128(a, w4);
    row[0] = w4[0];
    row[1] = w4[1];
    row[2] = w4[2];
    row[3] = w4[3];
    if (off + b > PFOR_WORD_BITS) {
      load128(a + 16u, w4);
      row[4] = w4[0];
      row[5] = w4[1];
      row[6] = w4[2];
      row[7] = w4[3];
    }
  }
  uint mask = low_mask(b);
  [ForceUnroll]
  for (uint c = 0; c < CPT; ++c)
    v[c] = ((body_word(row, c) |
             (body_word(row, c + 16u / PFOR_BYTES) << PFOR_WORD_BITS)) >>
            off) &
           mask;
  if (flags != 0) {
    [ForceUnroll]
    for (uint c = 0; c < CPT; ++c) {
      uint index = rank + countbits(flags & low_mask(c));
      if (((flags >> c) & 1u) != 0 && index < nexc) {
        uint bitpos = index * eb;
        uint a = excStart + (bitpos >> 3);
        v[c] |= ((load32(a) >> (bitpos & 7u)) & low_mask(eb)) << b;
      }
    }
  }
}

void block_planar(uint t, uint h, inout uint v[CPT]) {
  [ForceUnroll]
  for (uint c = 1; c < CPT; ++c)
    v[c] += v[c - 1];
  if (h == 16u) {
    uint left = wave_xor(v[CPT - 1], 16u);
    if (t >= 16u)
      [ForceUnroll]
      for (uint c = 0; c < CPT; ++c)
        v[c] += left;
    __target_switch {
    case metal:
      [ForceUnroll]
      for (uint c = 0; c < CPT; ++c) {
        uint p = wave_inclusive(v[c]);
        uint top = wave_broadcast(p, 15u);
        v[c] = t >= 16u ? p - top : p;
      }
    default:
      uint i = t & 15u;
      [ForceUnroll]
      for (uint d = 1u; d < 16u; d *= 2u) {
        [ForceUnroll]
        for (uint c = 0; c < CPT; ++c) {
          uint u = wave_up(v[c], d);
          if (i >= d)
            v[c] += u;
        }
      }
    }
    return;
  }
  uint total = v[CPT - 1];
  uint inclusive = total;
  for (uint d = h; d < WAVE; d *= 2u) {
    uint u = wave_up(inclusive, d);
    if (t >= d)
      inclusive += u;
  }
  uint before = inclusive - total;
  [ForceUnroll]
  for (uint c = 0; c < CPT; ++c)
    v[c] += before;
  uint i = t % h;
  for (uint d = 1u; d < h; d *= 2u) {
    [ForceUnroll]
    for (uint c = 0; c < CPT; ++c) {
      uint u = wave_up(v[c], d);
      if (i >= d)
        v[c] += u;
    }
  }
}

void store_block_row(uint elt, uint n, uint v[CPT]) {
  if (n == CPT && aligned_run(elt)) {
    store_run(elt, v);
    return;
  }
  [ForceUnroll]
  for (uint c = 0; c < CPT; ++c)
    if (c < n)
      output[elt + c] = Elt(v[c]);
}

void decode_strip(uint index, uint t) {
  strip_task strip = strips[index];
  if (strip.blockCount == 0)
    return;
  uint h = strip.height;
  uint blockWidth = SEGMENT / h;
  uint i = t % h;
  uint x = (t / h) * CPT;
  bool bad = false;
  uint carry = 0;
  for (uint k = 0; k < strip.blockCount; ++k) {
    block_ref ref = bandBlocks[strip.firstBlock + k];
    uint v[CPT];
    if (ref.rest == BAD_BLOCK) {
      [ForceUnroll]
      for (uint c = 0; c < CPT; ++c)
        v[c] = 0;
    } else {
      block_lane(ref.hdr, ref.rest, t, v, bad);
      [ForceUnroll]
      for (uint c = 0; c < CPT; ++c)
        v[c] = unzigzag(v[c]);
    }
    block_planar(t, h, v);
    [ForceUnroll]
    for (uint c = 0; c < CPT; ++c)
      v[c] += carry;
    carry = wave_at(v[CPT - 1], i + WAVE - h);
    uint x0 = k * blockWidth;
    uint valid = min(blockWidth, strip.columns - x0);
    if (x < valid)
      store_block_row(strip.outBase + i * strip.width + x0 + x,
                      min(CPT, valid - x), v);
  }
  if (WaveActiveAnyTrue(bad) && t == 0)
    report(STATUS_POSITIONS, strip.frame);
}
