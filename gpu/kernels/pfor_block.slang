// One PFOR block per wave. Lane t decodes values 8t..8t+7.

static const uint PFOR_WORD_BITS = 8 * PFOR_BYTES;
static const uint PFOR_LANES = 16 / PFOR_BYTES;

uint body_word(uint row[8], uint c) {
#if PFOR_BYTES == 1
  return (row[c >> 2] >> ((c & 3u) * 8u)) & 0xFFu;
#else
  return (row[c >> 1] >> ((c & 1u) * 16u)) & 0xFFFFu;
#endif
}

void pfor_lane(uint base, uint t, out uint v[CPT]) {
  uint b = load8(base);
  uint nexc = load8(base + 1u);

  uint bitmap[8];
  [ForceUnroll]
  for (uint w = 0; w < 8; ++w)
    bitmap[w] = 0;
  uint eb = 0;
  uint excStart = base + 2u;
  uint body = base + 2u;
  if (nexc != 0) {
    uint ebMode = load8(base + 2u);
    eb = ebMode & 0x7Fu;
    uint posStart = base + 3u;
    if ((ebMode >> 7) == 0) {
      uint mine = (t < 8u) ? load32(posStart + t * 4u) : 0u;
      [ForceUnroll]
      for (uint w = 0; w < 8; ++w)
        bitmap[w] = WaveReadLaneAt(mine, w);
      excStart = posStart + 32u;
    } else {
      uint p = (t < nexc) ? load8(posStart + t) : 0xFFFFu;
      [ForceUnroll]
      for (uint w = 0; w < 8; ++w)
        bitmap[w] = WaveActiveBitOr((p >> 5) == w ? (1u << (p & 31u)) : 0u);
      excStart = posStart + nexc;
    }
    body = excStart + ((nexc * eb + 7u) >> 3);
  }

  uint i0 = t * CPT;
  uint flags = (bitmap[t >> 2] >> ((t & 3u) * 8u)) & 0xFFu;
  uint rank = 0;
  [ForceUnroll]
  for (uint w = 0; w < 8; ++w)
    if (w < (t >> 2))
      rank += countbits(bitmap[w]);
  rank += countbits(bitmap[t >> 2] & low_mask((t & 3u) * 8u));

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
    uint w[4];
    load128(a, w);
    row[0] = w[0];
    row[1] = w[1];
    row[2] = w[2];
    row[3] = w[3];
    if (off + b > PFOR_WORD_BITS) {
      load128(a + 16u, w);
      row[4] = w[0];
      row[5] = w[1];
      row[6] = w[2];
      row[7] = w[3];
    }
  }
  uint mask = low_mask(b);
  [ForceUnroll]
  for (uint c = 0; c < CPT; ++c)
    v[c] = ((body_word(row, c) |
             (body_word(row, c + 16u / PFOR_BYTES) << PFOR_WORD_BITS)) >>
            off) &
           mask;

  [ForceUnroll]
  for (uint c = 0; c < CPT; ++c) {
    if (((flags >> c) & 1u) != 0) {
      uint bitpos = (rank + countbits(flags & low_mask(c))) * eb;
      uint a = excStart + (bitpos >> 3);
      v[c] |= ((load32(a) >> (bitpos & 7u)) & low_mask(eb)) << b;
    }
  }
}
