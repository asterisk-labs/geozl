// OpenZL PivCo Huffman lane decoder. The host supplies the tree and bitmap
// offsets; build_ranks creates the rank directory used by decode_symbols.

static const uint PIVCO_BYTES = 8;
static const uint MAX_DEPTH = 16;

struct tree_node {
  uint links;
  uint leaf;
};

struct node_bits {
  uint offset;
  uint count;
  uint ranks;
};

struct pivco_block {
  uint nodeBase;
  uint treeBase;
  uint symbolBase;
  uint outBase;
  uint length;
  uint nodeCount;
};

ByteAddressBuffer symbols;
StructuredBuffer<tree_node> trees;
StructuredBuffer<node_bits> nodes;
StructuredBuffer<uint> internalNodes;
StructuredBuffer<pivco_block> pivcoBlocks;
RWStructuredBuffer<uint2> ranks;
RWByteAddressBuffer lanes;

[shader("compute")]
[numthreads(WAVES_PER_GROUP * WAVE, 1, 1)]
void build_ranks(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
  uint lane = thread.x % WAVE;
  uint index = internalNodes[group.x * WAVES_PER_GROUP + thread.x / WAVE];
  if (index == 0xFFFFFFFFu)
    return;
  node_bits n = nodes[index];
  uint words = (n.count + 31u) >> 5;
  uint carry = 0;
  for (uint w0 = 0; w0 < words; w0 += WAVE) {
    uint w = w0 + lane;
    uint ones = 0;
    uint bits = 0;
    if (w < words) {
      bits = load32(n.offset + w * 4u);
      if (w == words - 1u)
        bits &= low_mask(n.count - w * 32u);
      ones = countbits(bits);
    }
    uint before = WavePrefixSum(ones) + carry;
    if (w < words)
      ranks[n.ranks + w] = uint2(before, bits);
    carry += WaveActiveSum(ones);
  }
}

uint symbol_at(uint symbolBase, uint rank) {
  uint a = symbolBase + rank;
  return (symbols.Load(a & ~3u) >> ((a & 3u) * 8u)) & 0xFFu;
}

static const uint MAX_NODES = 511;
groupshared uint treeLinks[MAX_NODES];
groupshared uint treeLeaf[MAX_NODES];
groupshared uint nodeOffset[MAX_NODES];
groupshared uint nodeRanks[MAX_NODES];

void walk_run(pivco_block blk, uint i0, out uint packed[PIVCO_BYTES / 4]) {
  uint node[PIVCO_BYTES];
  uint pos[PIVCO_BYTES];
  [ForceUnroll]
  for (uint c = 0; c < PIVCO_BYTES; ++c) {
    node[c] = 0;
    pos[c] = min(i0 + c, blk.length - 1u);
  }
  for (uint step = 0; step < MAX_DEPTH; ++step) {
    bool moved = false;
    [ForceUnroll]
    for (uint c = 0; c < PIVCO_BYTES; ++c) {
      if ((treeLeaf[node[c]] >> 16) == 0) {
        uint2 r = ranks[nodeRanks[node[c]] + (pos[c] >> 5)];
        uint onesBefore = r.x + countbits(r.y & low_mask(pos[c] & 31u));
        uint links = treeLinks[node[c]];
        bool right = ((r.y >> (pos[c] & 31u)) & 1u) != 0;
        pos[c] = right ? onesBefore : pos[c] - onesBefore;
        node[c] = right ? (links >> 16) : (links & 0xFFFFu);
        moved = true;
      }
    }
    if (!moved)
      break;
  }
  [ForceUnroll]
  for (uint w = 0; w < PIVCO_BYTES / 4; ++w)
    packed[w] = 0;
  [ForceUnroll]
  for (uint c = 0; c < PIVCO_BYTES; ++c) {
    uint leaf = treeLeaf[node[c]];
    uint depth = (leaf >> 8) & 0xFFu;
    uint idx = 0;
    if (depth != 0) {
      uint bitpos = pos[c] * depth;
      idx = (load32(nodeOffset[node[c]] + (bitpos >> 3)) >>
             (bitpos & 7u)) &
            low_mask(depth);
    }
    packed[c >> 2] |=
        symbol_at(blk.symbolBase, (leaf & 0xFFu) + idx) << ((c & 3u) * 8u);
  }
}

[shader("compute")]
[numthreads(WAVES_PER_GROUP * WAVE, 1, 1)]
void decode_symbols(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
  pivco_block blk = pivcoBlocks[group.x];
  for (uint n = thread.x; n < blk.nodeCount; n += WAVES_PER_GROUP * WAVE) {
    tree_node tn = trees[blk.treeBase + n];
    node_bits nb = nodes[blk.nodeBase + n];
    treeLinks[n] = tn.links;
    treeLeaf[n] = tn.leaf;
    nodeOffset[n] = nb.offset;
    nodeRanks[n] = nb.ranks;
  }
  GroupMemoryBarrierWithGroupSync();
  uint i0 = (group.y * WAVES_PER_GROUP * WAVE + thread.x) * PIVCO_BYTES;
  if (i0 >= blk.length)
    return;
  uint packed[PIVCO_BYTES / 4];
  walk_run(blk, i0, packed);
  [ForceUnroll]
  for (uint w = 0; w < PIVCO_BYTES / 4; ++w)
    lanes.Store(blk.outBase + i0 + w * 4u, packed[w]);
}
