// OpenZL PivCo Huffman lane decoder. One workgroup owns a block and rebuilds
// its tree bottom-up. Even levels write to lanes; odd levels use the output
// allocation as temporary space before join_lanes writes the final samples.

static const uint PIVCO_BYTES = 8;
// OpenZL caps the PivCo Huffman table log, and therefore the tree depth, at 12.
static const uint MAX_DEPTH = 12;

struct tree_node {
  uint links;
  uint leaf;
};

struct node_bits {
  uint offset;
  uint count;
  uint output;
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
StructuredBuffer<pivco_block> pivcoBlocks;
RWByteAddressBuffer lanes;
RWStructuredBuffer<uint8_t> laneBytes;

uint symbol_at(uint symbolBase, uint rank) {
  uint a = symbolBase + rank;
  return (symbols.Load(a & ~3u) >> ((a & 3u) * 8u)) & 0xFFu;
}

static const uint MAX_NODES = 511;
groupshared uint treeLinks[MAX_NODES];
groupshared uint treeLeaf[MAX_NODES];
groupshared uint nodeOffset[MAX_NODES];
groupshared uint nodeCount[MAX_NODES];
groupshared uint nodeOutput[MAX_NODES];

uint read_parity(bool fromLanes, uint offset) {
  return fromLanes ? uint(laneBytes[offset]) : uint(scratch[offset]);
}

void write_parity(bool toLanes, uint offset, uint value) {
  if (toLanes)
    laneBytes[offset] = uint8_t(value);
  else
    scratch[offset] = uint8_t(value);
}

void decode_leaf(pivco_block blk, uint node, uint leaf, uint count,
                 uint outputBase, bool toLanes, uint lane) {
  uint flatDepth = (leaf >> 8) & 0xFFu;
  for (uint p0 = lane * PIVCO_BYTES; p0 < count;
       p0 += WAVE * PIVCO_BYTES) {
    [ForceUnroll]
    for (uint c = 0; c < PIVCO_BYTES; ++c) {
      uint p = p0 + c;
      if (p < count) {
        uint index = 0;
        if (flatDepth != 0) {
          uint bitpos = p * flatDepth;
          index = (load32(nodeOffset[node] + (bitpos >> 3)) >>
                   (bitpos & 7u)) &
                  low_mask(flatDepth);
        }
        write_parity(toLanes, outputBase + p,
                     symbol_at(blk.symbolBase, (leaf & 0xFFu) + index));
      }
    }
  }
}

void merge_byte(uint bits, uint p0, uint count, uint leftCount,
                uint onesBefore, uint zerosBefore, uint outputBase,
                bool toLanes) {
  uint localOnes = 0;
  uint localZeros = 0;
  [ForceUnroll]
  for (uint c = 0; c < PIVCO_BYTES; ++c) {
    uint p = p0 + c;
    if (p < count) {
      bool right = ((bits >> c) & 1u) != 0;
      uint rightSource = leftCount + onesBefore + localOnes;
      uint leftSource = zerosBefore + localZeros;
      uint source = outputBase + select(right, rightSource, leftSource);
      write_parity(toLanes, outputBase + p,
                   read_parity(!toLanes, source));
      localOnes += right ? 1u : 0u;
      localZeros += right ? 0u : 1u;
    }
  }
}

void decode_node(pivco_block blk, uint node, uint lane) {
  uint leaf = treeLeaf[node];
  uint depth = leaf >> 24;
  uint count = nodeCount[node];
  uint outputBase = nodeOutput[node];
  bool toLanes = (depth & 1u) == 0;

  if ((leaf & (1u << 16)) != 0) {
    decode_leaf(blk, node, leaf, count, outputBase, toLanes, lane);
    return;
  }

  uint links = treeLinks[node];
  uint left = links & 0xFFFFu;
  uint leftCount = nodeCount[left];
  uint carry = 0;
  uint bytes = (count + PIVCO_BYTES - 1u) / PIVCO_BYTES;
  uint chunks = (bytes + WAVE - 1u) / WAVE;
  for (uint chunk = 0; chunk < chunks; ++chunk) {
    uint byteIndex = chunk * WAVE + lane;
    uint p0 = byteIndex * PIVCO_BYTES;
    uint safeByteIndex = min(byteIndex, bytes - 1u);
    uint bits = load32(nodeOffset[node] + safeByteIndex) & 0xFFu;
    uint remaining = count - min(p0, count);
    bits &= low_mask(min(PIVCO_BYTES, remaining));
    bits = select(byteIndex < bytes, bits, 0u);
    uint ones = countbits(bits);
    uint onesBefore = carry + WavePrefixSum(ones);
    uint zerosBefore = p0 - onesBefore;
    merge_byte(bits, p0, count, leftCount, onesBefore, zerosBefore,
               outputBase, toLanes);
    carry += WaveActiveSum(ones);
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
    nodeCount[n] = nb.count;
    nodeOutput[n] = nb.output;
  }
  GroupMemoryBarrierWithGroupSync();

  uint wave = thread.x / WAVE;
  uint lane = thread.x % WAVE;
  for (int depth = int(MAX_DEPTH); depth >= 0; --depth) {
    for (uint first = 0; first < blk.nodeCount; first += WAVES_PER_GROUP) {
      uint node = first + wave;
      if (node < blk.nodeCount && int(treeLeaf[node] >> 24) == depth)
        decode_node(blk, node, lane);
    }
    DeviceMemoryBarrierWithGroupSync();
  }
}
