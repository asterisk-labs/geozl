#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct pivco_block_0
{
    uint nodeBase_0;
    uint treeBase_0;
    uint symbolBase_0;
    uint outBase_0;
    uint length_0;
    uint nodeCount_0;
};

struct tree_node_0
{
    uint links_0;
    uint leaf_0;
};

struct node_bits_0
{
    uint offset_0;
    uint count_0;
    uint output_0;
};

struct lane_block_0
{
    uint payOff_0;
    uint outBase_1;
};

struct join_task_0
{
    uint laneBase_0;
    uint laneStride_0;
    uint count_1;
    uint outBase_2;
    uint start_0;
    uint transform_0;
};

struct plane_task_0
{
    uint outBase_3;
    uint width_0;
    uint rows_0;
};

struct raw_copy_0
{
    uint source_0;
    uint destination_0;
    uint count_2;
};

struct KernelContext_0
{
    pivco_block_0 device* pivcoBlocks_0;
    tree_node_0 device* trees_0;
    node_bits_0 device* nodes_0;
    uint32_t device* payload_0;
    uint32_t device* symbols_0;
    uchar device* laneBytes_0;
    uchar device* scratch_0;
    lane_block_0 device* laneBlocks_0;
    uint32_t device* lanes_0;
    join_task_0 device* joins_0;
    uint32_t device* outputBytes_0;
    ushort device* output_1;
    plane_task_0 device* planes_0;
    raw_copy_0 device* rawCopies_0;
    ushort device* rawValues_0;
    array<uint, int(511)> threadgroup* treeLinks_0;
    array<uint, int(511)> threadgroup* treeLeaf_0;
    array<uint, int(511)> threadgroup* nodeOffset_0;
    array<uint, int(511)> threadgroup* nodeCount_1;
    array<uint, int(511)> threadgroup* nodeOutput_0;
};

uint load32_0(uint a_0, KernelContext_0 thread* kernelContext_0)
{
    uint shift_0 = (a_0 & 3U) * 8U;
    uint _S1 = a_0 & 4294967292U;
    uint _S2 = as_type<uint>(kernelContext_0->payload_0[(uint(int(_S1)))>>2]);
    if(shift_0 == 0U)
    {
        return _S2;
    }
    uint _S3 = as_type<uint>(kernelContext_0->payload_0[(uint(int(_S1 + 4U)))>>2]);
    return (_S2 >> shift_0) | (_S3 << (32U - shift_0));
}

uint low_mask_0(uint b_0)
{
    uint _S4;
    if(b_0 >= 32U)
    {
        _S4 = 4294967295U;
    }
    else
    {
        _S4 = (1U << b_0) - 1U;
    }
    return _S4;
}

uint symbol_at_0(uint symbolBase_1, uint rank_0, KernelContext_0 thread* kernelContext_1)
{
    uint a_1 = symbolBase_1 + rank_0;
    uint _S5 = as_type<uint>(kernelContext_1->symbols_0[(uint(int(a_1 & 4294967292U)))>>2]);
    return (_S5 >> ((a_1 & 3U) * 8U)) & 255U;
}

void write_parity_0(bool toLanes_0, uint offset_1, uint value_0, KernelContext_0 thread* kernelContext_2)
{
    if(toLanes_0)
    {
        *(kernelContext_2->laneBytes_0+offset_1) = uchar(value_0);
    }
    else
    {
        *(kernelContext_2->scratch_0+offset_1) = uchar(value_0);
    }
    return;
}

void write_parity_1(bool toLanes_1, uint offset_2, uint value_1, KernelContext_0 thread* kernelContext_3)
{
    if(toLanes_1)
    {
        *(kernelContext_3->laneBytes_0+offset_2) = uchar(value_1);
    }
    else
    {
        *(kernelContext_3->scratch_0+offset_2) = uchar(value_1);
    }
    return;
}

void decode_leaf_0(const pivco_block_0 thread* blk_0, uint node_0, uint leaf_1, uint count_3, uint outputBase_0, bool toLanes_2, uint lane_0, KernelContext_0 thread* kernelContext_4)
{
    uint _S6 = (leaf_1 >> 8U) & 255U;
    uint p0_0 = lane_0 * 8U;
    for(;;)
    {
        bool _S7 = p0_0 < count_3;
        if(_S7)
        {
        }
        else
        {
            break;
        }
        uint index_0;
        if(_S7)
        {
            if(_S6 != 0U)
            {
                uint bitpos_0 = p0_0 * _S6;
                uint _S8 = load32_0((*kernelContext_4->nodeOffset_0)[node_0] + (bitpos_0 >> 3U), kernelContext_4);
                index_0 = (_S8 >> (bitpos_0 & 7U)) & (low_mask_0(_S6));
            }
            else
            {
                index_0 = 0U;
            }
            uint _S9 = outputBase_0 + p0_0;
            uint _S10 = symbol_at_0(blk_0->symbolBase_0, (leaf_1 & 255U) + index_0, kernelContext_4);
            write_parity_0(toLanes_2, _S9, _S10, kernelContext_4);
        }
        uint p_0 = p0_0 + 1U;
        if(p_0 < count_3)
        {
            if(_S6 != 0U)
            {
                uint bitpos_1 = p_0 * _S6;
                uint _S11 = load32_0((*kernelContext_4->nodeOffset_0)[node_0] + (bitpos_1 >> 3U), kernelContext_4);
                index_0 = (_S11 >> (bitpos_1 & 7U)) & (low_mask_0(_S6));
            }
            else
            {
                index_0 = 0U;
            }
            uint _S12 = outputBase_0 + p_0;
            uint _S13 = symbol_at_0(blk_0->symbolBase_0, (leaf_1 & 255U) + index_0, kernelContext_4);
            write_parity_0(toLanes_2, _S12, _S13, kernelContext_4);
        }
        uint p_1 = p0_0 + 2U;
        if(p_1 < count_3)
        {
            if(_S6 != 0U)
            {
                uint bitpos_2 = p_1 * _S6;
                uint _S14 = load32_0((*kernelContext_4->nodeOffset_0)[node_0] + (bitpos_2 >> 3U), kernelContext_4);
                index_0 = (_S14 >> (bitpos_2 & 7U)) & (low_mask_0(_S6));
            }
            else
            {
                index_0 = 0U;
            }
            uint _S15 = outputBase_0 + p_1;
            uint _S16 = symbol_at_0(blk_0->symbolBase_0, (leaf_1 & 255U) + index_0, kernelContext_4);
            write_parity_0(toLanes_2, _S15, _S16, kernelContext_4);
        }
        uint p_2 = p0_0 + 3U;
        if(p_2 < count_3)
        {
            if(_S6 != 0U)
            {
                uint bitpos_3 = p_2 * _S6;
                uint _S17 = load32_0((*kernelContext_4->nodeOffset_0)[node_0] + (bitpos_3 >> 3U), kernelContext_4);
                index_0 = (_S17 >> (bitpos_3 & 7U)) & (low_mask_0(_S6));
            }
            else
            {
                index_0 = 0U;
            }
            uint _S18 = outputBase_0 + p_2;
            uint _S19 = symbol_at_0(blk_0->symbolBase_0, (leaf_1 & 255U) + index_0, kernelContext_4);
            write_parity_0(toLanes_2, _S18, _S19, kernelContext_4);
        }
        uint p_3 = p0_0 + 4U;
        if(p_3 < count_3)
        {
            if(_S6 != 0U)
            {
                uint bitpos_4 = p_3 * _S6;
                uint _S20 = load32_0((*kernelContext_4->nodeOffset_0)[node_0] + (bitpos_4 >> 3U), kernelContext_4);
                index_0 = (_S20 >> (bitpos_4 & 7U)) & (low_mask_0(_S6));
            }
            else
            {
                index_0 = 0U;
            }
            uint _S21 = outputBase_0 + p_3;
            uint _S22 = symbol_at_0(blk_0->symbolBase_0, (leaf_1 & 255U) + index_0, kernelContext_4);
            write_parity_0(toLanes_2, _S21, _S22, kernelContext_4);
        }
        uint p_4 = p0_0 + 5U;
        if(p_4 < count_3)
        {
            if(_S6 != 0U)
            {
                uint bitpos_5 = p_4 * _S6;
                uint _S23 = load32_0((*kernelContext_4->nodeOffset_0)[node_0] + (bitpos_5 >> 3U), kernelContext_4);
                index_0 = (_S23 >> (bitpos_5 & 7U)) & (low_mask_0(_S6));
            }
            else
            {
                index_0 = 0U;
            }
            uint _S24 = outputBase_0 + p_4;
            uint _S25 = symbol_at_0(blk_0->symbolBase_0, (leaf_1 & 255U) + index_0, kernelContext_4);
            write_parity_0(toLanes_2, _S24, _S25, kernelContext_4);
        }
        uint p_5 = p0_0 + 6U;
        if(p_5 < count_3)
        {
            if(_S6 != 0U)
            {
                uint bitpos_6 = p_5 * _S6;
                uint _S26 = load32_0((*kernelContext_4->nodeOffset_0)[node_0] + (bitpos_6 >> 3U), kernelContext_4);
                index_0 = (_S26 >> (bitpos_6 & 7U)) & (low_mask_0(_S6));
            }
            else
            {
                index_0 = 0U;
            }
            uint _S27 = outputBase_0 + p_5;
            uint _S28 = symbol_at_0(blk_0->symbolBase_0, (leaf_1 & 255U) + index_0, kernelContext_4);
            write_parity_0(toLanes_2, _S27, _S28, kernelContext_4);
        }
        uint p_6 = p0_0 + 7U;
        if(p_6 < count_3)
        {
            if(_S6 != 0U)
            {
                uint bitpos_7 = p_6 * _S6;
                uint _S29 = load32_0((*kernelContext_4->nodeOffset_0)[node_0] + (bitpos_7 >> 3U), kernelContext_4);
                index_0 = (_S29 >> (bitpos_7 & 7U)) & (low_mask_0(_S6));
            }
            else
            {
                index_0 = 0U;
            }
            uint _S30 = outputBase_0 + p_6;
            uint _S31 = symbol_at_0(blk_0->symbolBase_0, (leaf_1 & 255U) + index_0, kernelContext_4);
            write_parity_0(toLanes_2, _S30, _S31, kernelContext_4);
        }
        p0_0 = p0_0 + 256U;
    }
    return;
}

uint read_parity_0(bool fromLanes_0, uint offset_3, KernelContext_0 thread* kernelContext_5)
{
    uint _S32;
    if(fromLanes_0)
    {
        _S32 = uint(*(kernelContext_5->laneBytes_0+offset_3));
    }
    else
    {
        _S32 = uint(*(kernelContext_5->scratch_0+offset_3));
    }
    return _S32;
}

void merge_byte_0(uint bits_0, uint p0_1, uint count_4, uint leftCount_0, uint onesBefore_0, uint zerosBefore_0, uint outputBase_1, bool toLanes_3, KernelContext_0 thread* kernelContext_6)
{
    uint localOnes_0;
    uint localZeros_0;
    if(p0_1 < count_4)
    {
        bool right_0 = ((bits_0 >> 0U) & 1U) != 0U;
        uint _S33 = outputBase_1 + p0_1;
        uint _S34 = read_parity_0(!toLanes_3, outputBase_1 + select(zerosBefore_0, leftCount_0 + onesBefore_0, right_0), kernelContext_6);
        write_parity_1(toLanes_3, _S33, _S34, kernelContext_6);
        if(right_0)
        {
            localOnes_0 = 1U;
        }
        else
        {
            localOnes_0 = 0U;
        }
        if(right_0)
        {
            localZeros_0 = 0U;
        }
        else
        {
            localZeros_0 = 1U;
        }
    }
    else
    {
        localOnes_0 = 0U;
        localZeros_0 = 0U;
    }
    uint p_7 = p0_1 + 1U;
    uint _S35;
    if(p_7 < count_4)
    {
        bool right_1 = ((bits_0 >> 1U) & 1U) != 0U;
        uint _S36 = outputBase_1 + p_7;
        uint _S37 = read_parity_0(!toLanes_3, outputBase_1 + select(zerosBefore_0 + localZeros_0, leftCount_0 + onesBefore_0 + localOnes_0, right_1), kernelContext_6);
        write_parity_1(toLanes_3, _S36, _S37, kernelContext_6);
        if(right_1)
        {
            _S35 = 1U;
        }
        else
        {
            _S35 = 0U;
        }
        uint localOnes_1 = localOnes_0 + _S35;
        if(right_1)
        {
            localOnes_0 = 0U;
        }
        else
        {
            localOnes_0 = 1U;
        }
        uint localZeros_1 = localZeros_0 + localOnes_0;
        localOnes_0 = localOnes_1;
        localZeros_0 = localZeros_1;
    }
    uint p_8 = p0_1 + 2U;
    if(p_8 < count_4)
    {
        bool right_2 = ((bits_0 >> 2U) & 1U) != 0U;
        uint _S38 = outputBase_1 + p_8;
        uint _S39 = read_parity_0(!toLanes_3, outputBase_1 + select(zerosBefore_0 + localZeros_0, leftCount_0 + onesBefore_0 + localOnes_0, right_2), kernelContext_6);
        write_parity_1(toLanes_3, _S38, _S39, kernelContext_6);
        if(right_2)
        {
            _S35 = 1U;
        }
        else
        {
            _S35 = 0U;
        }
        uint localOnes_2 = localOnes_0 + _S35;
        if(right_2)
        {
            localOnes_0 = 0U;
        }
        else
        {
            localOnes_0 = 1U;
        }
        uint localZeros_2 = localZeros_0 + localOnes_0;
        localOnes_0 = localOnes_2;
        localZeros_0 = localZeros_2;
    }
    uint p_9 = p0_1 + 3U;
    if(p_9 < count_4)
    {
        bool right_3 = ((bits_0 >> 3U) & 1U) != 0U;
        uint _S40 = outputBase_1 + p_9;
        uint _S41 = read_parity_0(!toLanes_3, outputBase_1 + select(zerosBefore_0 + localZeros_0, leftCount_0 + onesBefore_0 + localOnes_0, right_3), kernelContext_6);
        write_parity_1(toLanes_3, _S40, _S41, kernelContext_6);
        if(right_3)
        {
            _S35 = 1U;
        }
        else
        {
            _S35 = 0U;
        }
        uint localOnes_3 = localOnes_0 + _S35;
        if(right_3)
        {
            localOnes_0 = 0U;
        }
        else
        {
            localOnes_0 = 1U;
        }
        uint localZeros_3 = localZeros_0 + localOnes_0;
        localOnes_0 = localOnes_3;
        localZeros_0 = localZeros_3;
    }
    uint p_10 = p0_1 + 4U;
    if(p_10 < count_4)
    {
        bool right_4 = ((bits_0 >> 4U) & 1U) != 0U;
        uint _S42 = outputBase_1 + p_10;
        uint _S43 = read_parity_0(!toLanes_3, outputBase_1 + select(zerosBefore_0 + localZeros_0, leftCount_0 + onesBefore_0 + localOnes_0, right_4), kernelContext_6);
        write_parity_1(toLanes_3, _S42, _S43, kernelContext_6);
        if(right_4)
        {
            _S35 = 1U;
        }
        else
        {
            _S35 = 0U;
        }
        uint localOnes_4 = localOnes_0 + _S35;
        if(right_4)
        {
            localOnes_0 = 0U;
        }
        else
        {
            localOnes_0 = 1U;
        }
        uint localZeros_4 = localZeros_0 + localOnes_0;
        localOnes_0 = localOnes_4;
        localZeros_0 = localZeros_4;
    }
    uint p_11 = p0_1 + 5U;
    if(p_11 < count_4)
    {
        bool right_5 = ((bits_0 >> 5U) & 1U) != 0U;
        uint _S44 = outputBase_1 + p_11;
        uint _S45 = read_parity_0(!toLanes_3, outputBase_1 + select(zerosBefore_0 + localZeros_0, leftCount_0 + onesBefore_0 + localOnes_0, right_5), kernelContext_6);
        write_parity_1(toLanes_3, _S44, _S45, kernelContext_6);
        if(right_5)
        {
            _S35 = 1U;
        }
        else
        {
            _S35 = 0U;
        }
        uint localOnes_5 = localOnes_0 + _S35;
        if(right_5)
        {
            localOnes_0 = 0U;
        }
        else
        {
            localOnes_0 = 1U;
        }
        uint localZeros_5 = localZeros_0 + localOnes_0;
        localOnes_0 = localOnes_5;
        localZeros_0 = localZeros_5;
    }
    uint p_12 = p0_1 + 6U;
    if(p_12 < count_4)
    {
        bool right_6 = ((bits_0 >> 6U) & 1U) != 0U;
        uint _S46 = outputBase_1 + p_12;
        uint _S47 = read_parity_0(!toLanes_3, outputBase_1 + select(zerosBefore_0 + localZeros_0, leftCount_0 + onesBefore_0 + localOnes_0, right_6), kernelContext_6);
        write_parity_1(toLanes_3, _S46, _S47, kernelContext_6);
        if(right_6)
        {
            _S35 = 1U;
        }
        else
        {
            _S35 = 0U;
        }
        uint localOnes_6 = localOnes_0 + _S35;
        if(right_6)
        {
            localOnes_0 = 0U;
        }
        else
        {
            localOnes_0 = 1U;
        }
        uint localZeros_6 = localZeros_0 + localOnes_0;
        localOnes_0 = localOnes_6;
        localZeros_0 = localZeros_6;
    }
    uint p_13 = p0_1 + 7U;
    if(p_13 < count_4)
    {
        uint _S48 = outputBase_1 + p_13;
        uint _S49 = read_parity_0(!toLanes_3, outputBase_1 + select(zerosBefore_0 + localZeros_0, leftCount_0 + onesBefore_0 + localOnes_0, ((bits_0 >> 7U) & 1U) != 0U), kernelContext_6);
        write_parity_1(toLanes_3, _S48, _S49, kernelContext_6);
    }
    return;
}

void decode_node_0(const pivco_block_0 thread* blk_1, uint node_1, uint lane_1, KernelContext_0 thread* kernelContext_7)
{
    uint leaf_2 = (*kernelContext_7->treeLeaf_0)[node_1];
    uint count_5 = (*kernelContext_7->nodeCount_1)[node_1];
    uint outputBase_2 = (*kernelContext_7->nodeOutput_0)[node_1];
    bool toLanes_4 = ((((*kernelContext_7->treeLeaf_0)[node_1]) >> 24U) & 1U) == 0U;
    if((((*kernelContext_7->treeLeaf_0)[node_1]) & 65536U) != 0U)
    {
        decode_leaf_0(blk_1, node_1, leaf_2, count_5, outputBase_2, toLanes_4, lane_1, kernelContext_7);
        return;
    }
    uint _S50 = (*kernelContext_7->nodeCount_1)[((*kernelContext_7->treeLinks_0)[node_1]) & 65535U];
    uint bytes_0 = (count_5 + 8U - 1U) / 8U;
    uint _S51 = (bytes_0 + 32U - 1U) / 32U;
    uint chunk_0 = 0U;
    uint carry_0 = 0U;
    for(;;)
    {
        if(chunk_0 < _S51)
        {
        }
        else
        {
            break;
        }
        uint byteIndex_0 = chunk_0 * 32U + lane_1;
        uint p0_2 = byteIndex_0 * 8U;
        uint _S52 = load32_0((*kernelContext_7->nodeOffset_0)[node_1] + min(byteIndex_0, bytes_0 - 1U), kernelContext_7);
        uint bits_1 = select(0U, (_S52 & 255U) & (low_mask_0(min(8U, count_5 - min(p0_2, count_5)))), byteIndex_0 < bytes_0);
        uint ones_0 = ((uint)popcount((bits_1)));
        uint _S53 = simd_prefix_exclusive_sum(ones_0);
        uint onesBefore_1 = carry_0 + _S53;
        merge_byte_0(bits_1, p0_2, count_5, _S50, onesBefore_1, p0_2 - onesBefore_1, outputBase_2, toLanes_4, kernelContext_7);
        uint _S54 = simd_sum(ones_0);
        uint carry_1 = carry_0 + _S54;
        chunk_0 = chunk_0 + 1U;
        carry_0 = carry_1;
    }
    return;
}

[[kernel]] void decode_symbols(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_0 [[thread_position_in_threadgroup]], pivco_block_0 device* pivcoBlocks_1 [[buffer(4)]], tree_node_0 device* trees_1 [[buffer(2)]], node_bits_0 device* nodes_1 [[buffer(3)]], uint32_t device* payload_1 [[buffer(0)]], uint32_t device* symbols_1 [[buffer(1)]], uchar device* laneBytes_1 [[buffer(6)]], uchar device* scratch_1 [[buffer(12)]], lane_block_0 device* laneBlocks_1 [[buffer(7)]], uint32_t device* lanes_1 [[buffer(5)]], join_task_0 device* joins_1 [[buffer(8)]], uint32_t device* outputBytes_1 [[buffer(11)]], ushort device* output_2 [[buffer(10)]], plane_task_0 device* planes_1 [[buffer(9)]], raw_copy_0 device* rawCopies_1 [[buffer(14)]], ushort device* rawValues_1 [[buffer(13)]])
{
    uint _S55;
    thread KernelContext_0 kernelContext_8;
    (&kernelContext_8)->pivcoBlocks_0 = pivcoBlocks_1;
    (&kernelContext_8)->trees_0 = trees_1;
    (&kernelContext_8)->nodes_0 = nodes_1;
    (&kernelContext_8)->payload_0 = payload_1;
    (&kernelContext_8)->symbols_0 = symbols_1;
    (&kernelContext_8)->laneBytes_0 = laneBytes_1;
    (&kernelContext_8)->scratch_0 = scratch_1;
    (&kernelContext_8)->laneBlocks_0 = laneBlocks_1;
    (&kernelContext_8)->lanes_0 = lanes_1;
    (&kernelContext_8)->joins_0 = joins_1;
    (&kernelContext_8)->outputBytes_0 = outputBytes_1;
    (&kernelContext_8)->output_1 = output_2;
    (&kernelContext_8)->planes_0 = planes_1;
    (&kernelContext_8)->rawCopies_0 = rawCopies_1;
    (&kernelContext_8)->rawValues_0 = rawValues_1;
    threadgroup array<uint, int(511)> treeLinks_1;
    (&kernelContext_8)->treeLinks_0 = &treeLinks_1;
    threadgroup array<uint, int(511)> treeLeaf_1;
    (&kernelContext_8)->treeLeaf_0 = &treeLeaf_1;
    threadgroup array<uint, int(511)> nodeOffset_1;
    (&kernelContext_8)->nodeOffset_0 = &nodeOffset_1;
    threadgroup array<uint, int(511)> nodeCount_2;
    (&kernelContext_8)->nodeCount_1 = &nodeCount_2;
    threadgroup array<uint, int(511)> nodeOutput_1;
    (&kernelContext_8)->nodeOutput_0 = &nodeOutput_1;
    pivco_block_0 _S56 = pivcoBlocks_1[group_0.x];
    uint _S57 = thread_0.x;
    uint n_0 = _S57;
    for(;;)
    {
        _S55 = _S56.nodeCount_0;
        if(n_0 < (_S56.nodeCount_0))
        {
        }
        else
        {
            break;
        }
        tree_node_0 tn_0 = (&kernelContext_8)->trees_0[_S56.treeBase_0 + n_0];
        node_bits_0 nb_0 = (&kernelContext_8)->nodes_0[_S56.nodeBase_0 + n_0];
        (*(&kernelContext_8)->treeLinks_0)[n_0] = tn_0.links_0;
        (*(&kernelContext_8)->treeLeaf_0)[n_0] = tn_0.leaf_0;
        (*(&kernelContext_8)->nodeOffset_0)[n_0] = nb_0.offset_0;
        (*(&kernelContext_8)->nodeCount_1)[n_0] = nb_0.count_0;
        (*(&kernelContext_8)->nodeOutput_0)[n_0] = nb_0.output_0;
        n_0 = n_0 + 128U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint _S58 = _S57 / 32U;
    uint _S59 = _S57 % 32U;
    int depth_0 = int(12);
    for(;;)
    {
        if(depth_0 >= int(0))
        {
        }
        else
        {
            break;
        }
        uint first_0 = 0U;
        for(;;)
        {
            if(first_0 < _S55)
            {
            }
            else
            {
                break;
            }
            uint node_2 = first_0 + _S58;
            bool _S60;
            if(node_2 < _S55)
            {
                _S60 = int(((*(&kernelContext_8)->treeLeaf_0)[node_2]) >> 24U) == depth_0;
            }
            else
            {
                _S60 = false;
            }
            if(_S60)
            {
                thread pivco_block_0 _S61 = _S56;
                decode_node_0(&_S61, node_2, _S59, &kernelContext_8);
            }
            first_0 = first_0 + 4U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
        depth_0 = depth_0 - int(1);
    }
    return;
}

uint load8_0(uint a_2, KernelContext_0 thread* kernelContext_9)
{
    uint _S62 = as_type<uint>(kernelContext_9->payload_0[(uint(int(a_2 & 4294967292U)))>>2]);
    return (_S62 >> ((a_2 & 3U) * 8U)) & 255U;
}

void load128_0(uint a_3, array<uint, int(4)> thread* w_0, KernelContext_0 thread* kernelContext_10)
{
    uint base_0 = a_3 & 4294967292U;
    uint shift_1 = (a_3 & 3U) * 8U;
    uint _S63 = as_type<uint>(kernelContext_10->payload_0[(base_0)>>2]);
    uint _S64 = as_type<uint>(kernelContext_10->payload_0[(base_0 + 4U)>>2]);
    uint _S65 = as_type<uint>(kernelContext_10->payload_0[(base_0 + 8U)>>2]);
    uint _S66 = as_type<uint>(kernelContext_10->payload_0[(base_0 + 12U)>>2]);
    uint _S67 = as_type<uint>(kernelContext_10->payload_0[(uint(int(base_0 + 16U)))>>2]);
    if(shift_1 == 0U)
    {
        (*w_0)[int(0)] = _S63;
        (*w_0)[int(1)] = _S64;
        (*w_0)[int(2)] = _S65;
        (*w_0)[int(3)] = _S66;
        return;
    }
    uint _S68 = 32U - shift_1;
    (*w_0)[int(0)] = (_S63 >> shift_1) | (_S64 << _S68);
    (*w_0)[int(1)] = (_S64 >> shift_1) | (_S65 << _S68);
    (*w_0)[int(2)] = (_S65 >> shift_1) | (_S66 << _S68);
    (*w_0)[int(3)] = (_S66 >> shift_1) | (_S67 << _S68);
    return;
}

uint body_word_0(const array<uint, int(8)> thread* row_0, uint c_0)
{
    return (((*row_0)[c_0 >> 2U]) >> ((c_0 & 3U) * 8U)) & 255U;
}

void pfor_lane_0(uint base_1, uint t_0, array<uint, int(8)> thread* v_0, KernelContext_0 thread* kernelContext_11)
{
    uint _S69 = load8_0(base_1, kernelContext_11);
    uint _S70 = load8_0(base_1 + 1U, kernelContext_11);
    thread array<uint, int(8)> bitmap_0;
    bitmap_0[0U] = 0U;
    bitmap_0[1U] = 0U;
    bitmap_0[2U] = 0U;
    bitmap_0[3U] = 0U;
    bitmap_0[4U] = 0U;
    bitmap_0[5U] = 0U;
    bitmap_0[6U] = 0U;
    bitmap_0[7U] = 0U;
    uint _S71 = base_1 + 2U;
    uint body_0;
    uint eb_0;
    uint excStart_0;
    uint rank_1;
    uint off_0;
    if(_S70 != 0U)
    {
        uint _S72 = load8_0(_S71, kernelContext_11);
        uint eb_1 = _S72 & 127U;
        uint posStart_0 = base_1 + 3U;
        if((_S72 >> 7U) == 0U)
        {
            if(t_0 < 8U)
            {
                uint _S73 = load32_0(posStart_0 + t_0 * 4U, kernelContext_11);
                body_0 = _S73;
            }
            else
            {
                body_0 = 0U;
            }
            uint _S74 = (simd_shuffle((body_0), ushort((int(0)))));
            bitmap_0[0U] = _S74;
            uint _S75 = (simd_shuffle((body_0), ushort((int(1)))));
            bitmap_0[1U] = _S75;
            uint _S76 = (simd_shuffle((body_0), ushort((int(2)))));
            bitmap_0[2U] = _S76;
            uint _S77 = (simd_shuffle((body_0), ushort((int(3)))));
            bitmap_0[3U] = _S77;
            uint _S78 = (simd_shuffle((body_0), ushort((int(4)))));
            bitmap_0[4U] = _S78;
            uint _S79 = (simd_shuffle((body_0), ushort((int(5)))));
            bitmap_0[5U] = _S79;
            uint _S80 = (simd_shuffle((body_0), ushort((int(6)))));
            bitmap_0[6U] = _S80;
            uint _S81 = (simd_shuffle((body_0), ushort((int(7)))));
            bitmap_0[7U] = _S81;
            body_0 = posStart_0 + 32U;
        }
        else
        {
            uint _S82 = (_S70 + 32U - 1U) / 32U;
            body_0 = 0U;
            for(;;)
            {
                if(body_0 < _S82)
                {
                }
                else
                {
                    break;
                }
                uint i_0 = body_0 * 32U + t_0;
                if(i_0 < _S70)
                {
                    uint _S83 = load8_0(posStart_0 + i_0, kernelContext_11);
                    eb_0 = _S83;
                }
                else
                {
                    eb_0 = 65535U;
                }
                uint _S84 = eb_0 >> 5U;
                if(_S84 == 0U)
                {
                    excStart_0 = 1U << (eb_0 & 31U);
                }
                else
                {
                    excStart_0 = 0U;
                }
                uint _S85 = simd_or(excStart_0);
                bitmap_0[0U] = (bitmap_0[0U]) | _S85;
                if(_S84 == 1U)
                {
                    rank_1 = 1U << (eb_0 & 31U);
                }
                else
                {
                    rank_1 = 0U;
                }
                uint _S86 = simd_or(rank_1);
                bitmap_0[1U] = (bitmap_0[1U]) | _S86;
                if(_S84 == 2U)
                {
                    off_0 = 1U << (eb_0 & 31U);
                }
                else
                {
                    off_0 = 0U;
                }
                uint _S87 = simd_or(off_0);
                bitmap_0[2U] = (bitmap_0[2U]) | _S87;
                uint _S88;
                if(_S84 == 3U)
                {
                    _S88 = 1U << (eb_0 & 31U);
                }
                else
                {
                    _S88 = 0U;
                }
                uint _S89 = simd_or(_S88);
                bitmap_0[3U] = (bitmap_0[3U]) | _S89;
                uint _S90;
                if(_S84 == 4U)
                {
                    _S90 = 1U << (eb_0 & 31U);
                }
                else
                {
                    _S90 = 0U;
                }
                uint _S91 = simd_or(_S90);
                bitmap_0[4U] = (bitmap_0[4U]) | _S91;
                uint _S92;
                if(_S84 == 5U)
                {
                    _S92 = 1U << (eb_0 & 31U);
                }
                else
                {
                    _S92 = 0U;
                }
                uint _S93 = simd_or(_S92);
                bitmap_0[5U] = (bitmap_0[5U]) | _S93;
                uint _S94;
                if(_S84 == 6U)
                {
                    _S94 = 1U << (eb_0 & 31U);
                }
                else
                {
                    _S94 = 0U;
                }
                uint _S95 = simd_or(_S94);
                bitmap_0[6U] = (bitmap_0[6U]) | _S95;
                uint _S96;
                if(_S84 == 7U)
                {
                    _S96 = 1U << (eb_0 & 31U);
                }
                else
                {
                    _S96 = 0U;
                }
                uint _S97 = simd_or(_S96);
                bitmap_0[7U] = (bitmap_0[7U]) | _S97;
                body_0 = body_0 + 1U;
            }
            body_0 = posStart_0 + _S70;
        }
        uint _S98 = body_0;
        body_0 = body_0 + ((_S70 * eb_1 + 7U) >> 3U);
        eb_0 = eb_1;
        excStart_0 = _S98;
    }
    else
    {
        body_0 = _S71;
        eb_0 = 0U;
        excStart_0 = _S71;
    }
    uint i0_0 = t_0 * 8U;
    uint _S99 = t_0 >> 2U;
    uint _S100 = (t_0 & 3U) * 8U;
    uint _S101 = ((bitmap_0[_S99]) >> _S100) & 255U;
    if(0U < _S99)
    {
        rank_1 = ((uint)popcount((bitmap_0[0U])));
    }
    else
    {
        rank_1 = 0U;
    }
    if(1U < _S99)
    {
        rank_1 = rank_1 + ((uint)popcount((bitmap_0[1U])));
    }
    if(2U < _S99)
    {
        rank_1 = rank_1 + ((uint)popcount((bitmap_0[2U])));
    }
    if(3U < _S99)
    {
        rank_1 = rank_1 + ((uint)popcount((bitmap_0[3U])));
    }
    if(4U < _S99)
    {
        rank_1 = rank_1 + ((uint)popcount((bitmap_0[4U])));
    }
    if(5U < _S99)
    {
        rank_1 = rank_1 + ((uint)popcount((bitmap_0[5U])));
    }
    if(6U < _S99)
    {
        rank_1 = rank_1 + ((uint)popcount((bitmap_0[6U])));
    }
    if(7U < _S99)
    {
        rank_1 = rank_1 + ((uint)popcount((bitmap_0[7U])));
    }
    uint rank_2 = rank_1 + ((uint)popcount(((bitmap_0[_S99]) & (low_mask_0(_S100)))));
    thread array<uint, int(8)> row_1;
    row_1[0U] = 0U;
    row_1[1U] = 0U;
    row_1[2U] = 0U;
    row_1[3U] = 0U;
    row_1[4U] = 0U;
    row_1[5U] = 0U;
    row_1[6U] = 0U;
    row_1[7U] = 0U;
    if(_S69 != 0U)
    {
        uint bitpos_8 = i0_0 / 16U * _S69;
        uint off_1 = bitpos_8 % 8U;
        uint a_4 = body_0 + bitpos_8 / 8U * 16U + i0_0 % 16U;
        thread array<uint, int(4)> w_1;
        load128_0(a_4, &w_1, kernelContext_11);
        row_1[int(0)] = w_1[int(0)];
        row_1[int(1)] = w_1[int(1)];
        row_1[int(2)] = w_1[int(2)];
        row_1[int(3)] = w_1[int(3)];
        if((off_1 + _S69) > 8U)
        {
            load128_0(a_4 + 16U, &w_1, kernelContext_11);
            row_1[int(4)] = w_1[int(0)];
            row_1[int(5)] = w_1[int(1)];
            row_1[int(6)] = w_1[int(2)];
            row_1[int(7)] = w_1[int(3)];
        }
        off_0 = off_1;
    }
    else
    {
        off_0 = 0U;
    }
    uint _S102 = low_mask_0(_S69);
    thread array<uint, int(8)> _S103 = row_1;
    uint _S104 = body_word_0(&_S103, 0U);
    thread array<uint, int(8)> _S105 = row_1;
    uint _S106 = body_word_0(&_S105, 16U);
    (*v_0)[0U] = ((_S104 | (_S106 << 8U)) >> off_0) & _S102;
    thread array<uint, int(8)> _S107 = row_1;
    uint _S108 = body_word_0(&_S107, 1U);
    thread array<uint, int(8)> _S109 = row_1;
    uint _S110 = body_word_0(&_S109, 17U);
    (*v_0)[1U] = ((_S108 | (_S110 << 8U)) >> off_0) & _S102;
    thread array<uint, int(8)> _S111 = row_1;
    uint _S112 = body_word_0(&_S111, 2U);
    thread array<uint, int(8)> _S113 = row_1;
    uint _S114 = body_word_0(&_S113, 18U);
    (*v_0)[2U] = ((_S112 | (_S114 << 8U)) >> off_0) & _S102;
    thread array<uint, int(8)> _S115 = row_1;
    uint _S116 = body_word_0(&_S115, 3U);
    thread array<uint, int(8)> _S117 = row_1;
    uint _S118 = body_word_0(&_S117, 19U);
    (*v_0)[3U] = ((_S116 | (_S118 << 8U)) >> off_0) & _S102;
    thread array<uint, int(8)> _S119 = row_1;
    uint _S120 = body_word_0(&_S119, 4U);
    thread array<uint, int(8)> _S121 = row_1;
    uint _S122 = body_word_0(&_S121, 20U);
    (*v_0)[4U] = ((_S120 | (_S122 << 8U)) >> off_0) & _S102;
    thread array<uint, int(8)> _S123 = row_1;
    uint _S124 = body_word_0(&_S123, 5U);
    thread array<uint, int(8)> _S125 = row_1;
    uint _S126 = body_word_0(&_S125, 21U);
    (*v_0)[5U] = ((_S124 | (_S126 << 8U)) >> off_0) & _S102;
    thread array<uint, int(8)> _S127 = row_1;
    uint _S128 = body_word_0(&_S127, 6U);
    thread array<uint, int(8)> _S129 = row_1;
    uint _S130 = body_word_0(&_S129, 22U);
    (*v_0)[6U] = ((_S128 | (_S130 << 8U)) >> off_0) & _S102;
    thread array<uint, int(8)> _S131 = row_1;
    uint _S132 = body_word_0(&_S131, 7U);
    thread array<uint, int(8)> _S133 = row_1;
    uint _S134 = body_word_0(&_S133, 23U);
    (*v_0)[7U] = ((_S132 | (_S134 << 8U)) >> off_0) & _S102;
    if(((_S101 >> 0U) & 1U) != 0U)
    {
        uint bitpos_9 = (rank_2 + ((uint)popcount((_S101 & (low_mask_0(0U)))))) * eb_0;
        uint _S135 = load32_0(excStart_0 + (bitpos_9 >> 3U), kernelContext_11);
        (*v_0)[0U] = ((*v_0)[0U]) | (((_S135 >> (bitpos_9 & 7U)) & (low_mask_0(eb_0))) << _S69);
    }
    if(((_S101 >> 1U) & 1U) != 0U)
    {
        uint bitpos_10 = (rank_2 + ((uint)popcount((_S101 & (low_mask_0(1U)))))) * eb_0;
        uint _S136 = load32_0(excStart_0 + (bitpos_10 >> 3U), kernelContext_11);
        (*v_0)[1U] = ((*v_0)[1U]) | (((_S136 >> (bitpos_10 & 7U)) & (low_mask_0(eb_0))) << _S69);
    }
    if(((_S101 >> 2U) & 1U) != 0U)
    {
        uint bitpos_11 = (rank_2 + ((uint)popcount((_S101 & (low_mask_0(2U)))))) * eb_0;
        uint _S137 = load32_0(excStart_0 + (bitpos_11 >> 3U), kernelContext_11);
        (*v_0)[2U] = ((*v_0)[2U]) | (((_S137 >> (bitpos_11 & 7U)) & (low_mask_0(eb_0))) << _S69);
    }
    if(((_S101 >> 3U) & 1U) != 0U)
    {
        uint bitpos_12 = (rank_2 + ((uint)popcount((_S101 & (low_mask_0(3U)))))) * eb_0;
        uint _S138 = load32_0(excStart_0 + (bitpos_12 >> 3U), kernelContext_11);
        (*v_0)[3U] = ((*v_0)[3U]) | (((_S138 >> (bitpos_12 & 7U)) & (low_mask_0(eb_0))) << _S69);
    }
    if(((_S101 >> 4U) & 1U) != 0U)
    {
        uint bitpos_13 = (rank_2 + ((uint)popcount((_S101 & (low_mask_0(4U)))))) * eb_0;
        uint _S139 = load32_0(excStart_0 + (bitpos_13 >> 3U), kernelContext_11);
        (*v_0)[4U] = ((*v_0)[4U]) | (((_S139 >> (bitpos_13 & 7U)) & (low_mask_0(eb_0))) << _S69);
    }
    if(((_S101 >> 5U) & 1U) != 0U)
    {
        uint bitpos_14 = (rank_2 + ((uint)popcount((_S101 & (low_mask_0(5U)))))) * eb_0;
        uint _S140 = load32_0(excStart_0 + (bitpos_14 >> 3U), kernelContext_11);
        (*v_0)[5U] = ((*v_0)[5U]) | (((_S140 >> (bitpos_14 & 7U)) & (low_mask_0(eb_0))) << _S69);
    }
    if(((_S101 >> 6U) & 1U) != 0U)
    {
        uint bitpos_15 = (rank_2 + ((uint)popcount((_S101 & (low_mask_0(6U)))))) * eb_0;
        uint _S141 = load32_0(excStart_0 + (bitpos_15 >> 3U), kernelContext_11);
        (*v_0)[6U] = ((*v_0)[6U]) | (((_S141 >> (bitpos_15 & 7U)) & (low_mask_0(eb_0))) << _S69);
    }
    if(((_S101 >> 7U) & 1U) != 0U)
    {
        uint bitpos_16 = (rank_2 + ((uint)popcount((_S101 & (low_mask_0(7U)))))) * eb_0;
        uint _S142 = load32_0(excStart_0 + (bitpos_16 >> 3U), kernelContext_11);
        (*v_0)[7U] = ((*v_0)[7U]) | (((_S142 >> (bitpos_16 & 7U)) & (low_mask_0(eb_0))) << _S69);
    }
    return;
}

[[kernel]] void unpack_lane_blocks(uint3 group_1 [[threadgroup_position_in_grid]], uint3 thread_1 [[thread_position_in_threadgroup]], pivco_block_0 device* pivcoBlocks_2 [[buffer(4)]], tree_node_0 device* trees_2 [[buffer(2)]], node_bits_0 device* nodes_2 [[buffer(3)]], uint32_t device* payload_2 [[buffer(0)]], uint32_t device* symbols_2 [[buffer(1)]], uchar device* laneBytes_2 [[buffer(6)]], uchar device* scratch_2 [[buffer(12)]], lane_block_0 device* laneBlocks_2 [[buffer(7)]], uint32_t device* lanes_2 [[buffer(5)]], join_task_0 device* joins_2 [[buffer(8)]], uint32_t device* outputBytes_2 [[buffer(11)]], ushort device* output_3 [[buffer(10)]], plane_task_0 device* planes_2 [[buffer(9)]], raw_copy_0 device* rawCopies_2 [[buffer(14)]], ushort device* rawValues_2 [[buffer(13)]])
{
    thread KernelContext_0 kernelContext_12;
    (&kernelContext_12)->pivcoBlocks_0 = pivcoBlocks_2;
    (&kernelContext_12)->trees_0 = trees_2;
    (&kernelContext_12)->nodes_0 = nodes_2;
    (&kernelContext_12)->payload_0 = payload_2;
    (&kernelContext_12)->symbols_0 = symbols_2;
    (&kernelContext_12)->laneBytes_0 = laneBytes_2;
    (&kernelContext_12)->scratch_0 = scratch_2;
    (&kernelContext_12)->laneBlocks_0 = laneBlocks_2;
    (&kernelContext_12)->lanes_0 = lanes_2;
    (&kernelContext_12)->joins_0 = joins_2;
    (&kernelContext_12)->outputBytes_0 = outputBytes_2;
    (&kernelContext_12)->output_1 = output_3;
    (&kernelContext_12)->planes_0 = planes_2;
    (&kernelContext_12)->rawCopies_0 = rawCopies_2;
    (&kernelContext_12)->rawValues_0 = rawValues_2;
    threadgroup array<uint, int(511)> treeLinks_2;
    (&kernelContext_12)->treeLinks_0 = &treeLinks_2;
    threadgroup array<uint, int(511)> treeLeaf_2;
    (&kernelContext_12)->treeLeaf_0 = &treeLeaf_2;
    threadgroup array<uint, int(511)> nodeOffset_2;
    (&kernelContext_12)->nodeOffset_0 = &nodeOffset_2;
    threadgroup array<uint, int(511)> nodeCount_3;
    (&kernelContext_12)->nodeCount_1 = &nodeCount_3;
    threadgroup array<uint, int(511)> nodeOutput_2;
    (&kernelContext_12)->nodeOutput_0 = &nodeOutput_2;
    uint _S143 = thread_1.x;
    uint t_1 = _S143 % 32U;
    lane_block_0 blk_2 = laneBlocks_2[group_1.x * 4U + _S143 / 32U];
    thread array<uint, int(8)> v_1;
    pfor_lane_0(blk_2.payOff_0, t_1, &v_1, &kernelContext_12);
    uint _S144 = blk_2.outBase_1 + t_1 * 8U;
    uint _S145 = (((v_1[int(4)]) | ((v_1[int(5)]) << 8U)) | ((v_1[int(6)]) << 16U)) | ((v_1[int(7)]) << 24U);
    (&kernelContext_12)->lanes_0[(_S144)>>2] = as_type<uint32_t>((((v_1[int(0)]) | ((v_1[int(1)]) << 8U)) | ((v_1[int(2)]) << 16U)) | ((v_1[int(3)]) << 24U));
    (&kernelContext_12)->lanes_0[(_S144 + 4U)>>2] = as_type<uint32_t>(_S145);
    return;
}

uint unzigzag_0(uint v_2)
{
    return ((v_2 >> 1U) ^ (0U - (v_2 & 1U))) & (low_mask_0(16U));
}

bool aligned_run_0(uint elt_0)
{
    return ((elt_0 * 2U) & 15U) == 0U;
}

void store_run_0(uint elt_1, const array<uint, int(8)> thread* v_3, KernelContext_0 thread* kernelContext_13)
{
    uint _S146 = elt_1 * 2U;
    uint _S147 = (((*v_3)[int(2)]) & 65535U) | (((*v_3)[int(3)]) << 16U);
    uint _S148 = (((*v_3)[int(4)]) & 65535U) | (((*v_3)[int(5)]) << 16U);
    uint _S149 = (((*v_3)[int(6)]) & 65535U) | (((*v_3)[int(7)]) << 16U);
    kernelContext_13->outputBytes_0[(_S146)>>2] = as_type<uint32_t>((((*v_3)[int(0)]) & 65535U) | (((*v_3)[int(1)]) << 16U));
    kernelContext_13->outputBytes_0[(_S146 + 4U)>>2] = as_type<uint32_t>(_S147);
    kernelContext_13->outputBytes_0[(_S146 + 8U)>>2] = as_type<uint32_t>(_S148);
    kernelContext_13->outputBytes_0[(_S146 + 12U)>>2] = as_type<uint32_t>(_S149);
    return;
}

[[kernel]] void join_lanes(uint3 group_2 [[threadgroup_position_in_grid]], uint3 thread_2 [[thread_position_in_threadgroup]], pivco_block_0 device* pivcoBlocks_3 [[buffer(4)]], tree_node_0 device* trees_3 [[buffer(2)]], node_bits_0 device* nodes_3 [[buffer(3)]], uint32_t device* payload_3 [[buffer(0)]], uint32_t device* symbols_3 [[buffer(1)]], uchar device* laneBytes_3 [[buffer(6)]], uchar device* scratch_3 [[buffer(12)]], lane_block_0 device* laneBlocks_3 [[buffer(7)]], uint32_t device* lanes_3 [[buffer(5)]], join_task_0 device* joins_3 [[buffer(8)]], uint32_t device* outputBytes_3 [[buffer(11)]], ushort device* output_4 [[buffer(10)]], plane_task_0 device* planes_3 [[buffer(9)]], raw_copy_0 device* rawCopies_3 [[buffer(14)]], ushort device* rawValues_3 [[buffer(13)]])
{
    thread KernelContext_0 kernelContext_14;
    (&kernelContext_14)->pivcoBlocks_0 = pivcoBlocks_3;
    (&kernelContext_14)->trees_0 = trees_3;
    (&kernelContext_14)->nodes_0 = nodes_3;
    (&kernelContext_14)->payload_0 = payload_3;
    (&kernelContext_14)->symbols_0 = symbols_3;
    (&kernelContext_14)->laneBytes_0 = laneBytes_3;
    (&kernelContext_14)->scratch_0 = scratch_3;
    (&kernelContext_14)->laneBlocks_0 = laneBlocks_3;
    (&kernelContext_14)->lanes_0 = lanes_3;
    (&kernelContext_14)->joins_0 = joins_3;
    (&kernelContext_14)->outputBytes_0 = outputBytes_3;
    (&kernelContext_14)->output_1 = output_4;
    (&kernelContext_14)->planes_0 = planes_3;
    (&kernelContext_14)->rawCopies_0 = rawCopies_3;
    (&kernelContext_14)->rawValues_0 = rawValues_3;
    threadgroup array<uint, int(511)> treeLinks_3;
    (&kernelContext_14)->treeLinks_0 = &treeLinks_3;
    threadgroup array<uint, int(511)> treeLeaf_3;
    (&kernelContext_14)->treeLeaf_0 = &treeLeaf_3;
    threadgroup array<uint, int(511)> nodeOffset_3;
    (&kernelContext_14)->nodeOffset_0 = &nodeOffset_3;
    threadgroup array<uint, int(511)> nodeCount_4;
    (&kernelContext_14)->nodeCount_1 = &nodeCount_4;
    threadgroup array<uint, int(511)> nodeOutput_3;
    (&kernelContext_14)->nodeOutput_0 = &nodeOutput_3;
    join_task_0 f_0 = joins_3[group_2.x];
    uint i0_1 = f_0.start_0 + thread_2.x * 8U;
    if(i0_1 >= (f_0.count_1))
    {
        return;
    }
    thread array<uint, int(8)> v_4;
    v_4[0U] = 0U;
    v_4[1U] = 0U;
    v_4[2U] = 0U;
    v_4[3U] = 0U;
    v_4[4U] = 0U;
    v_4[5U] = 0U;
    v_4[6U] = 0U;
    v_4[7U] = 0U;
    uint _S150 = f_0.laneBase_0 + i0_1;
    uint _S151 = as_type<uint>((&kernelContext_14)->lanes_0[(_S150)>>2]);
    uint _S152 = as_type<uint>((&kernelContext_14)->lanes_0[(_S150 + 4U)>>2]);
    v_4[0U] = (v_4[0U]) | (((_S151 >> 0U) & 255U) << 0U);
    v_4[4U] = (v_4[4U]) | (((_S152 >> 0U) & 255U) << 0U);
    v_4[1U] = (v_4[1U]) | (((_S151 >> 8U) & 255U) << 0U);
    v_4[5U] = (v_4[5U]) | (((_S152 >> 8U) & 255U) << 0U);
    v_4[2U] = (v_4[2U]) | (((_S151 >> 16U) & 255U) << 0U);
    v_4[6U] = (v_4[6U]) | (((_S152 >> 16U) & 255U) << 0U);
    v_4[3U] = (v_4[3U]) | (((_S151 >> 24U) & 255U) << 0U);
    v_4[7U] = (v_4[7U]) | (((_S152 >> 24U) & 255U) << 0U);
    uint _S153 = f_0.laneBase_0 + f_0.laneStride_0 + i0_1;
    uint _S154 = as_type<uint>((&kernelContext_14)->lanes_0[(_S153)>>2]);
    uint _S155 = as_type<uint>((&kernelContext_14)->lanes_0[(_S153 + 4U)>>2]);
    v_4[0U] = (v_4[0U]) | (((_S154 >> 0U) & 255U) << 8U);
    v_4[4U] = (v_4[4U]) | (((_S155 >> 0U) & 255U) << 8U);
    v_4[1U] = (v_4[1U]) | (((_S154 >> 8U) & 255U) << 8U);
    v_4[5U] = (v_4[5U]) | (((_S155 >> 8U) & 255U) << 8U);
    v_4[2U] = (v_4[2U]) | (((_S154 >> 16U) & 255U) << 8U);
    v_4[6U] = (v_4[6U]) | (((_S155 >> 16U) & 255U) << 8U);
    v_4[3U] = (v_4[3U]) | (((_S154 >> 24U) & 255U) << 8U);
    v_4[7U] = (v_4[7U]) | (((_S155 >> 24U) & 255U) << 8U);
    if((f_0.transform_0) != 0U)
    {
        v_4[0U] = unzigzag_0(v_4[0U]);
        v_4[1U] = unzigzag_0(v_4[1U]);
        v_4[2U] = unzigzag_0(v_4[2U]);
        v_4[3U] = unzigzag_0(v_4[3U]);
        v_4[4U] = unzigzag_0(v_4[4U]);
        v_4[5U] = unzigzag_0(v_4[5U]);
        v_4[6U] = unzigzag_0(v_4[6U]);
        v_4[7U] = unzigzag_0(v_4[7U]);
    }
    uint elt_2 = f_0.outBase_2 + i0_1;
    bool _S156;
    if((i0_1 + 8U) <= (f_0.count_1))
    {
        _S156 = aligned_run_0(elt_2);
    }
    else
    {
        _S156 = false;
    }
    if(_S156)
    {
        thread array<uint, int(8)> _S157 = v_4;
        store_run_0(elt_2, &_S157, &kernelContext_14);
    }
    else
    {
        if(i0_1 < (f_0.count_1))
        {
            *((&kernelContext_14)->output_1+elt_2) = ushort(v_4[0U]);
        }
        if((i0_1 + 1U) < (f_0.count_1))
        {
            *((&kernelContext_14)->output_1+(elt_2 + 1U)) = ushort(v_4[1U]);
        }
        if((i0_1 + 2U) < (f_0.count_1))
        {
            *((&kernelContext_14)->output_1+(elt_2 + 2U)) = ushort(v_4[2U]);
        }
        if((i0_1 + 3U) < (f_0.count_1))
        {
            *((&kernelContext_14)->output_1+(elt_2 + 3U)) = ushort(v_4[3U]);
        }
        if((i0_1 + 4U) < (f_0.count_1))
        {
            *((&kernelContext_14)->output_1+(elt_2 + 4U)) = ushort(v_4[4U]);
        }
        if((i0_1 + 5U) < (f_0.count_1))
        {
            *((&kernelContext_14)->output_1+(elt_2 + 5U)) = ushort(v_4[5U]);
        }
        if((i0_1 + 6U) < (f_0.count_1))
        {
            *((&kernelContext_14)->output_1+(elt_2 + 6U)) = ushort(v_4[6U]);
        }
        if((i0_1 + 7U) < (f_0.count_1))
        {
            *((&kernelContext_14)->output_1+(elt_2 + 7U)) = ushort(v_4[7U]);
        }
    }
    return;
}

void load_run_0(uint elt_3, array<uint, int(8)> thread* v_5, KernelContext_0 thread* kernelContext_15)
{
    uint _S158 = elt_3 * 2U;
    uint _S159 = as_type<uint>(kernelContext_15->outputBytes_0[(_S158)>>2]);
    uint _S160 = as_type<uint>(kernelContext_15->outputBytes_0[(_S158 + 4U)>>2]);
    uint _S161 = as_type<uint>(kernelContext_15->outputBytes_0[(_S158 + 8U)>>2]);
    uint _S162 = as_type<uint>(kernelContext_15->outputBytes_0[(_S158 + 12U)>>2]);
    (*v_5)[int(0)] = _S159 & 65535U;
    (*v_5)[int(1)] = _S159 >> 16U;
    (*v_5)[int(2)] = _S160 & 65535U;
    (*v_5)[int(3)] = _S160 >> 16U;
    (*v_5)[int(4)] = _S161 & 65535U;
    (*v_5)[int(5)] = _S161 >> 16U;
    (*v_5)[int(6)] = _S162 & 65535U;
    (*v_5)[int(7)] = _S162 >> 16U;
    return;
}

void load_row_0(uint o0_0, uint x0_0, uint width_1, array<uint, int(8)> thread* r_0, KernelContext_0 thread* kernelContext_16)
{
    bool _S163;
    if((x0_0 + 8U) <= width_1)
    {
        _S163 = aligned_run_0(o0_0 + x0_0);
    }
    else
    {
        _S163 = false;
    }
    if(_S163)
    {
        load_run_0(o0_0 + x0_0, r_0, kernelContext_16);
    }
    else
    {
        uint _S164;
        if(x0_0 < width_1)
        {
            _S164 = uint(*(kernelContext_16->output_1+(o0_0 + x0_0)));
        }
        else
        {
            _S164 = 0U;
        }
        (*r_0)[0U] = _S164;
        if((x0_0 + 1U) < width_1)
        {
            _S164 = uint(*(kernelContext_16->output_1+(o0_0 + x0_0 + 1U)));
        }
        else
        {
            _S164 = 0U;
        }
        (*r_0)[1U] = _S164;
        if((x0_0 + 2U) < width_1)
        {
            _S164 = uint(*(kernelContext_16->output_1+(o0_0 + x0_0 + 2U)));
        }
        else
        {
            _S164 = 0U;
        }
        (*r_0)[2U] = _S164;
        if((x0_0 + 3U) < width_1)
        {
            _S164 = uint(*(kernelContext_16->output_1+(o0_0 + x0_0 + 3U)));
        }
        else
        {
            _S164 = 0U;
        }
        (*r_0)[3U] = _S164;
        if((x0_0 + 4U) < width_1)
        {
            _S164 = uint(*(kernelContext_16->output_1+(o0_0 + x0_0 + 4U)));
        }
        else
        {
            _S164 = 0U;
        }
        (*r_0)[4U] = _S164;
        if((x0_0 + 5U) < width_1)
        {
            _S164 = uint(*(kernelContext_16->output_1+(o0_0 + x0_0 + 5U)));
        }
        else
        {
            _S164 = 0U;
        }
        (*r_0)[5U] = _S164;
        if((x0_0 + 6U) < width_1)
        {
            _S164 = uint(*(kernelContext_16->output_1+(o0_0 + x0_0 + 6U)));
        }
        else
        {
            _S164 = 0U;
        }
        (*r_0)[6U] = _S164;
        if((x0_0 + 7U) < width_1)
        {
            _S164 = uint(*(kernelContext_16->output_1+(o0_0 + x0_0 + 7U)));
        }
        else
        {
            _S164 = 0U;
        }
        (*r_0)[7U] = _S164;
    }
    return;
}

void store_row_0(uint o0_1, uint x0_1, uint width_2, const array<uint, int(8)> thread* r_1, KernelContext_0 thread* kernelContext_17)
{
    bool _S165;
    if((x0_1 + 8U) <= width_2)
    {
        _S165 = aligned_run_0(o0_1 + x0_1);
    }
    else
    {
        _S165 = false;
    }
    if(_S165)
    {
        store_run_0(o0_1 + x0_1, r_1, kernelContext_17);
    }
    else
    {
        if(x0_1 < width_2)
        {
            *(kernelContext_17->output_1+(o0_1 + x0_1)) = ushort((*r_1)[0U]);
        }
        if((x0_1 + 1U) < width_2)
        {
            *(kernelContext_17->output_1+(o0_1 + x0_1 + 1U)) = ushort((*r_1)[1U]);
        }
        if((x0_1 + 2U) < width_2)
        {
            *(kernelContext_17->output_1+(o0_1 + x0_1 + 2U)) = ushort((*r_1)[2U]);
        }
        if((x0_1 + 3U) < width_2)
        {
            *(kernelContext_17->output_1+(o0_1 + x0_1 + 3U)) = ushort((*r_1)[3U]);
        }
        if((x0_1 + 4U) < width_2)
        {
            *(kernelContext_17->output_1+(o0_1 + x0_1 + 4U)) = ushort((*r_1)[4U]);
        }
        if((x0_1 + 5U) < width_2)
        {
            *(kernelContext_17->output_1+(o0_1 + x0_1 + 5U)) = ushort((*r_1)[5U]);
        }
        if((x0_1 + 6U) < width_2)
        {
            *(kernelContext_17->output_1+(o0_1 + x0_1 + 6U)) = ushort((*r_1)[6U]);
        }
        if((x0_1 + 7U) < width_2)
        {
            *(kernelContext_17->output_1+(o0_1 + x0_1 + 7U)) = ushort((*r_1)[7U]);
        }
    }
    return;
}

[[kernel]] void scan_planes(uint3 group_3 [[threadgroup_position_in_grid]], uint3 thread_3 [[thread_position_in_threadgroup]], pivco_block_0 device* pivcoBlocks_4 [[buffer(4)]], tree_node_0 device* trees_4 [[buffer(2)]], node_bits_0 device* nodes_4 [[buffer(3)]], uint32_t device* payload_4 [[buffer(0)]], uint32_t device* symbols_4 [[buffer(1)]], uchar device* laneBytes_4 [[buffer(6)]], uchar device* scratch_4 [[buffer(12)]], lane_block_0 device* laneBlocks_4 [[buffer(7)]], uint32_t device* lanes_4 [[buffer(5)]], join_task_0 device* joins_4 [[buffer(8)]], uint32_t device* outputBytes_4 [[buffer(11)]], ushort device* output_5 [[buffer(10)]], plane_task_0 device* planes_4 [[buffer(9)]], raw_copy_0 device* rawCopies_4 [[buffer(14)]], ushort device* rawValues_4 [[buffer(13)]])
{
    thread KernelContext_0 kernelContext_18;
    (&kernelContext_18)->pivcoBlocks_0 = pivcoBlocks_4;
    (&kernelContext_18)->trees_0 = trees_4;
    (&kernelContext_18)->nodes_0 = nodes_4;
    (&kernelContext_18)->payload_0 = payload_4;
    (&kernelContext_18)->symbols_0 = symbols_4;
    (&kernelContext_18)->laneBytes_0 = laneBytes_4;
    (&kernelContext_18)->scratch_0 = scratch_4;
    (&kernelContext_18)->laneBlocks_0 = laneBlocks_4;
    (&kernelContext_18)->lanes_0 = lanes_4;
    (&kernelContext_18)->joins_0 = joins_4;
    (&kernelContext_18)->outputBytes_0 = outputBytes_4;
    (&kernelContext_18)->output_1 = output_5;
    (&kernelContext_18)->planes_0 = planes_4;
    (&kernelContext_18)->rawCopies_0 = rawCopies_4;
    (&kernelContext_18)->rawValues_0 = rawValues_4;
    threadgroup array<uint, int(511)> treeLinks_4;
    (&kernelContext_18)->treeLinks_0 = &treeLinks_4;
    threadgroup array<uint, int(511)> treeLeaf_4;
    (&kernelContext_18)->treeLeaf_0 = &treeLeaf_4;
    threadgroup array<uint, int(511)> nodeOffset_4;
    (&kernelContext_18)->nodeOffset_0 = &nodeOffset_4;
    threadgroup array<uint, int(511)> nodeCount_5;
    (&kernelContext_18)->nodeCount_1 = &nodeCount_5;
    threadgroup array<uint, int(511)> nodeOutput_4;
    (&kernelContext_18)->nodeOutput_0 = &nodeOutput_4;
    uint _S166 = thread_3.x;
    uint t_2 = _S166 % 32U;
    plane_task_0 task_0 = planes_4[group_3.x * 4U + _S166 / 32U];
    bool _S167;
    if((task_0.width_0) > 256U)
    {
        _S167 = true;
    }
    else
    {
        _S167 = (task_0.rows_0) == 0U;
    }
    if(_S167)
    {
        return;
    }
    uint x0_2 = t_2 * 8U;
    thread array<uint, int(8)> column_0;
    thread array<uint, int(8)> next_0;
    column_0[0U] = 0U;
    column_0[1U] = 0U;
    column_0[2U] = 0U;
    column_0[3U] = 0U;
    column_0[4U] = 0U;
    column_0[5U] = 0U;
    column_0[6U] = 0U;
    column_0[7U] = 0U;
    load_row_0(task_0.outBase_3, x0_2, task_0.width_0, &next_0, &kernelContext_18);
    uint y_0 = 0U;
    for(;;)
    {
        if(y_0 < (task_0.rows_0))
        {
        }
        else
        {
            break;
        }
        uint o0_2 = task_0.outBase_3 + y_0 * task_0.width_0;
        thread array<uint, int(8)> r_2;
        r_2[0U] = next_0[0U];
        r_2[1U] = next_0[1U];
        r_2[2U] = next_0[2U];
        r_2[3U] = next_0[3U];
        r_2[4U] = next_0[4U];
        r_2[5U] = next_0[5U];
        r_2[6U] = next_0[6U];
        r_2[7U] = next_0[7U];
        uint _S168 = y_0 + 1U;
        if(_S168 < (task_0.rows_0))
        {
            load_row_0(o0_2 + task_0.width_0, x0_2, task_0.width_0, &next_0, &kernelContext_18);
        }
        uint local_0 = r_2[0U] + r_2[1U];
        uint local_1 = local_0 + r_2[2U];
        uint local_2 = local_1 + r_2[3U];
        uint local_3 = local_2 + r_2[4U];
        uint local_4 = local_3 + r_2[5U];
        uint local_5 = local_4 + r_2[6U];
        uint local_6 = local_5 + r_2[7U];
        uint _S169 = simd_prefix_exclusive_sum(local_6);
        uint _S170 = column_0[0U] + (r_2[0U] + _S169);
        column_0[0U] = _S170;
        r_2[0U] = _S170;
        uint _S171 = column_0[1U] + (local_0 + _S169);
        column_0[1U] = _S171;
        r_2[1U] = _S171;
        uint _S172 = column_0[2U] + (local_1 + _S169);
        column_0[2U] = _S172;
        r_2[2U] = _S172;
        uint _S173 = column_0[3U] + (local_2 + _S169);
        column_0[3U] = _S173;
        r_2[3U] = _S173;
        uint _S174 = column_0[4U] + (local_3 + _S169);
        column_0[4U] = _S174;
        r_2[4U] = _S174;
        uint _S175 = column_0[5U] + (local_4 + _S169);
        column_0[5U] = _S175;
        r_2[5U] = _S175;
        uint _S176 = column_0[6U] + (local_5 + _S169);
        column_0[6U] = _S176;
        r_2[6U] = _S176;
        uint _S177 = column_0[7U] + (local_6 + _S169);
        column_0[7U] = _S177;
        r_2[7U] = _S177;
        thread array<uint, int(8)> _S178 = r_2;
        store_row_0(o0_2, x0_2, task_0.width_0, &_S178, &kernelContext_18);
        y_0 = _S168;
    }
    return;
}

[[kernel]] void scan_rows(uint3 group_4 [[threadgroup_position_in_grid]], uint3 thread_4 [[thread_position_in_threadgroup]], pivco_block_0 device* pivcoBlocks_5 [[buffer(4)]], tree_node_0 device* trees_5 [[buffer(2)]], node_bits_0 device* nodes_5 [[buffer(3)]], uint32_t device* payload_5 [[buffer(0)]], uint32_t device* symbols_5 [[buffer(1)]], uchar device* laneBytes_5 [[buffer(6)]], uchar device* scratch_5 [[buffer(12)]], lane_block_0 device* laneBlocks_5 [[buffer(7)]], uint32_t device* lanes_5 [[buffer(5)]], join_task_0 device* joins_5 [[buffer(8)]], uint32_t device* outputBytes_5 [[buffer(11)]], ushort device* output_6 [[buffer(10)]], plane_task_0 device* planes_5 [[buffer(9)]], raw_copy_0 device* rawCopies_5 [[buffer(14)]], ushort device* rawValues_5 [[buffer(13)]])
{
    thread KernelContext_0 kernelContext_19;
    (&kernelContext_19)->pivcoBlocks_0 = pivcoBlocks_5;
    (&kernelContext_19)->trees_0 = trees_5;
    (&kernelContext_19)->nodes_0 = nodes_5;
    (&kernelContext_19)->payload_0 = payload_5;
    (&kernelContext_19)->symbols_0 = symbols_5;
    (&kernelContext_19)->laneBytes_0 = laneBytes_5;
    (&kernelContext_19)->scratch_0 = scratch_5;
    (&kernelContext_19)->laneBlocks_0 = laneBlocks_5;
    (&kernelContext_19)->lanes_0 = lanes_5;
    (&kernelContext_19)->joins_0 = joins_5;
    (&kernelContext_19)->outputBytes_0 = outputBytes_5;
    (&kernelContext_19)->output_1 = output_6;
    (&kernelContext_19)->planes_0 = planes_5;
    (&kernelContext_19)->rawCopies_0 = rawCopies_5;
    (&kernelContext_19)->rawValues_0 = rawValues_5;
    threadgroup array<uint, int(511)> treeLinks_5;
    (&kernelContext_19)->treeLinks_0 = &treeLinks_5;
    threadgroup array<uint, int(511)> treeLeaf_5;
    (&kernelContext_19)->treeLeaf_0 = &treeLeaf_5;
    threadgroup array<uint, int(511)> nodeOffset_5;
    (&kernelContext_19)->nodeOffset_0 = &nodeOffset_5;
    threadgroup array<uint, int(511)> nodeCount_6;
    (&kernelContext_19)->nodeCount_1 = &nodeCount_6;
    threadgroup array<uint, int(511)> nodeOutput_5;
    (&kernelContext_19)->nodeOutput_0 = &nodeOutput_5;
    uint _S179 = thread_4.x;
    uint _S180 = _S179 % 32U;
    plane_task_0 task_1 = planes_5[group_4.x];
    uint y_1 = group_4.y * 4U + _S179 / 32U;
    bool _S181;
    if((task_1.width_0) <= 256U)
    {
        _S181 = true;
    }
    else
    {
        _S181 = y_1 >= (task_1.rows_0);
    }
    if(_S181)
    {
        return;
    }
    uint _S182 = task_1.outBase_3 + y_1 * task_1.width_0;
    uint _S183 = (task_1.width_0 + 256U - 1U) / 256U;
    uint segment_0 = 0U;
    uint rowCarry_0 = 0U;
    for(;;)
    {
        if(segment_0 < _S183)
        {
        }
        else
        {
            break;
        }
        uint xs_0 = segment_0 * 256U + _S180 * 8U;
        thread array<uint, int(8)> r_3;
        load_row_0(_S182, xs_0, task_1.width_0, &r_3, &kernelContext_19);
        uint local_7 = r_3[0U] + r_3[1U];
        uint local_8 = local_7 + r_3[2U];
        uint local_9 = local_8 + r_3[3U];
        uint local_10 = local_9 + r_3[4U];
        uint local_11 = local_10 + r_3[5U];
        uint local_12 = local_11 + r_3[6U];
        uint local_13 = local_12 + r_3[7U];
        uint _S184 = simd_prefix_exclusive_sum(local_13);
        uint _S185 = _S184 + rowCarry_0;
        uint _S186 = simd_sum(local_13);
        uint rowCarry_1 = rowCarry_0 + _S186;
        r_3[0U] = r_3[0U] + _S185;
        r_3[1U] = local_7 + _S185;
        r_3[2U] = local_8 + _S185;
        r_3[3U] = local_9 + _S185;
        r_3[4U] = local_10 + _S185;
        r_3[5U] = local_11 + _S185;
        r_3[6U] = local_12 + _S185;
        r_3[7U] = local_13 + _S185;
        thread array<uint, int(8)> _S187 = r_3;
        store_row_0(_S182, xs_0, task_1.width_0, &_S187, &kernelContext_19);
        segment_0 = segment_0 + 1U;
        rowCarry_0 = rowCarry_1;
    }
    return;
}

[[kernel]] void scan_columns(uint3 group_5 [[threadgroup_position_in_grid]], uint3 thread_5 [[thread_position_in_threadgroup]], pivco_block_0 device* pivcoBlocks_6 [[buffer(4)]], tree_node_0 device* trees_6 [[buffer(2)]], node_bits_0 device* nodes_6 [[buffer(3)]], uint32_t device* payload_6 [[buffer(0)]], uint32_t device* symbols_6 [[buffer(1)]], uchar device* laneBytes_6 [[buffer(6)]], uchar device* scratch_6 [[buffer(12)]], lane_block_0 device* laneBlocks_6 [[buffer(7)]], uint32_t device* lanes_6 [[buffer(5)]], join_task_0 device* joins_6 [[buffer(8)]], uint32_t device* outputBytes_6 [[buffer(11)]], ushort device* output_7 [[buffer(10)]], plane_task_0 device* planes_6 [[buffer(9)]], raw_copy_0 device* rawCopies_6 [[buffer(14)]], ushort device* rawValues_6 [[buffer(13)]])
{
    thread KernelContext_0 kernelContext_20;
    (&kernelContext_20)->pivcoBlocks_0 = pivcoBlocks_6;
    (&kernelContext_20)->trees_0 = trees_6;
    (&kernelContext_20)->nodes_0 = nodes_6;
    (&kernelContext_20)->payload_0 = payload_6;
    (&kernelContext_20)->symbols_0 = symbols_6;
    (&kernelContext_20)->laneBytes_0 = laneBytes_6;
    (&kernelContext_20)->scratch_0 = scratch_6;
    (&kernelContext_20)->laneBlocks_0 = laneBlocks_6;
    (&kernelContext_20)->lanes_0 = lanes_6;
    (&kernelContext_20)->joins_0 = joins_6;
    (&kernelContext_20)->outputBytes_0 = outputBytes_6;
    (&kernelContext_20)->output_1 = output_7;
    (&kernelContext_20)->planes_0 = planes_6;
    (&kernelContext_20)->rawCopies_0 = rawCopies_6;
    (&kernelContext_20)->rawValues_0 = rawValues_6;
    threadgroup array<uint, int(511)> treeLinks_6;
    (&kernelContext_20)->treeLinks_0 = &treeLinks_6;
    threadgroup array<uint, int(511)> treeLeaf_6;
    (&kernelContext_20)->treeLeaf_0 = &treeLeaf_6;
    threadgroup array<uint, int(511)> nodeOffset_6;
    (&kernelContext_20)->nodeOffset_0 = &nodeOffset_6;
    threadgroup array<uint, int(511)> nodeCount_7;
    (&kernelContext_20)->nodeCount_1 = &nodeCount_7;
    threadgroup array<uint, int(511)> nodeOutput_6;
    (&kernelContext_20)->nodeOutput_0 = &nodeOutput_6;
    plane_task_0 task_2 = planes_6[group_5.x];
    uint x_0 = group_5.y * 128U + thread_5.x;
    bool _S188;
    if((task_2.width_0) <= 256U)
    {
        _S188 = true;
    }
    else
    {
        _S188 = x_0 >= (task_2.width_0);
    }
    if(_S188)
    {
        return;
    }
    uint y_2 = 0U;
    uint carry_2 = 0U;
    for(;;)
    {
        if(y_2 < (task_2.rows_0))
        {
        }
        else
        {
            break;
        }
        uint o_0 = task_2.outBase_3 + y_2 * task_2.width_0 + x_0;
        uint carry_3 = carry_2 + uint(*((&kernelContext_20)->output_1+o_0));
        *((&kernelContext_20)->output_1+o_0) = ushort(carry_3);
        y_2 = y_2 + 1U;
        carry_2 = carry_3;
    }
    return;
}

[[kernel]] void copy_raw(uint3 group_6 [[threadgroup_position_in_grid]], uint3 thread_6 [[thread_position_in_threadgroup]], pivco_block_0 device* pivcoBlocks_7 [[buffer(4)]], tree_node_0 device* trees_7 [[buffer(2)]], node_bits_0 device* nodes_7 [[buffer(3)]], uint32_t device* payload_7 [[buffer(0)]], uint32_t device* symbols_7 [[buffer(1)]], uchar device* laneBytes_7 [[buffer(6)]], uchar device* scratch_7 [[buffer(12)]], lane_block_0 device* laneBlocks_7 [[buffer(7)]], uint32_t device* lanes_7 [[buffer(5)]], join_task_0 device* joins_7 [[buffer(8)]], uint32_t device* outputBytes_7 [[buffer(11)]], ushort device* output_8 [[buffer(10)]], plane_task_0 device* planes_7 [[buffer(9)]], raw_copy_0 device* rawCopies_7 [[buffer(14)]], ushort device* rawValues_7 [[buffer(13)]])
{
    thread KernelContext_0 kernelContext_21;
    (&kernelContext_21)->pivcoBlocks_0 = pivcoBlocks_7;
    (&kernelContext_21)->trees_0 = trees_7;
    (&kernelContext_21)->nodes_0 = nodes_7;
    (&kernelContext_21)->payload_0 = payload_7;
    (&kernelContext_21)->symbols_0 = symbols_7;
    (&kernelContext_21)->laneBytes_0 = laneBytes_7;
    (&kernelContext_21)->scratch_0 = scratch_7;
    (&kernelContext_21)->laneBlocks_0 = laneBlocks_7;
    (&kernelContext_21)->lanes_0 = lanes_7;
    (&kernelContext_21)->joins_0 = joins_7;
    (&kernelContext_21)->outputBytes_0 = outputBytes_7;
    (&kernelContext_21)->output_1 = output_8;
    (&kernelContext_21)->planes_0 = planes_7;
    (&kernelContext_21)->rawCopies_0 = rawCopies_7;
    (&kernelContext_21)->rawValues_0 = rawValues_7;
    threadgroup array<uint, int(511)> treeLinks_7;
    (&kernelContext_21)->treeLinks_0 = &treeLinks_7;
    threadgroup array<uint, int(511)> treeLeaf_7;
    (&kernelContext_21)->treeLeaf_0 = &treeLeaf_7;
    threadgroup array<uint, int(511)> nodeOffset_7;
    (&kernelContext_21)->nodeOffset_0 = &nodeOffset_7;
    threadgroup array<uint, int(511)> nodeCount_8;
    (&kernelContext_21)->nodeCount_1 = &nodeCount_8;
    threadgroup array<uint, int(511)> nodeOutput_7;
    (&kernelContext_21)->nodeOutput_0 = &nodeOutput_7;
    raw_copy_0 _S189 = rawCopies_7[group_6.x];
    uint i_1 = thread_6.x;
    for(;;)
    {
        if(i_1 < (_S189.count_2))
        {
        }
        else
        {
            break;
        }
        *((&kernelContext_21)->output_1+(_S189.destination_0 + i_1)) = (&kernelContext_21)->rawValues_0[_S189.source_0 + i_1];
        i_1 = i_1 + 128U;
    }
    return;
}

