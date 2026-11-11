#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct block_task_0
{
    uint payOff_0;
    uint outBase_0;
    uint count_0;
    uint pad_0;
};

struct block_chunk_0
{
    uint frame_0;
    uint first_0;
    uint count_1;
    uint firstChunk_0;
};

struct band_frame_0
{
    uint payOff_1;
    uint paySize_0;
    uint blockCount_0;
    uint firstBlock_0;
    uint frame_1;
    uint pad0_0;
    uint pad1_0;
    uint pad2_0;
};

struct block_ref_0
{
    uint hdr_0;
    uint rest_0;
};

struct strip_task_0
{
    uint firstBlock_1;
    uint blockCount_1;
    uint outBase_1;
    uint width_0;
    uint height_0;
    uint frame_2;
    uint columns_0;
    uint pad_1;
};

struct raw_copy_0
{
    uint source_0;
    uint destination_0;
    uint count_2;
};

struct KernelContext_0
{
    block_task_0 device* blocks_0;
    uint32_t device* payload_0;
    uint32_t device* outputBytes_0;
    uchar device* output_0;
    block_chunk_0 device* blockChunks_0;
    band_frame_0 device* bandFrames_0;
    uint device* chunkSizes_0;
    block_ref_0 device* bandBlocks_0;
    uint device* status_0;
    strip_task_0 device* strips_0;
    raw_copy_0 device* rawCopies_0;
    uchar device* rawValues_0;
};

uint load8_0(uint a_0, KernelContext_0 thread* kernelContext_0)
{
    uint _S1 = as_type<uint>(kernelContext_0->payload_0[(uint(int(a_0 & 4294967292U)))>>2]);
    return (_S1 >> ((a_0 & 3U) * 8U)) & 255U;
}

uint load8_1(uint a_1, KernelContext_0 thread* kernelContext_1)
{
    uint _S2 = as_type<uint>(kernelContext_1->payload_0[(uint(int(a_1 & 4294967292U)))>>2]);
    return (_S2 >> ((a_1 & 3U) * 8U)) & 255U;
}

uint load32_0(uint a_2, KernelContext_0 thread* kernelContext_2)
{
    uint shift_0 = (a_2 & 3U) * 8U;
    uint _S3 = a_2 & 4294967292U;
    uint _S4 = as_type<uint>(kernelContext_2->payload_0[(uint(int(_S3)))>>2]);
    if(shift_0 == 0U)
    {
        return _S4;
    }
    uint _S5 = as_type<uint>(kernelContext_2->payload_0[(uint(int(_S3 + 4U)))>>2]);
    return (_S4 >> shift_0) | (_S5 << (32U - shift_0));
}

uint load32_1(uint a_3, KernelContext_0 thread* kernelContext_3)
{
    uint shift_1 = (a_3 & 3U) * 8U;
    uint _S6 = a_3 & 4294967292U;
    uint _S7 = as_type<uint>(kernelContext_3->payload_0[(uint(int(_S6)))>>2]);
    if(shift_1 == 0U)
    {
        return _S7;
    }
    uint _S8 = as_type<uint>(kernelContext_3->payload_0[(uint(int(_S6 + 4U)))>>2]);
    return (_S7 >> shift_1) | (_S8 << (32U - shift_1));
}

uint low_mask_0(uint b_0)
{
    uint _S9;
    if(b_0 >= 32U)
    {
        _S9 = 4294967295U;
    }
    else
    {
        _S9 = (1U << b_0) - 1U;
    }
    return _S9;
}

void load128_0(uint a_4, array<uint, int(4)> thread* w_0, KernelContext_0 thread* kernelContext_4)
{
    uint base_0 = a_4 & 4294967292U;
    uint shift_2 = (a_4 & 3U) * 8U;
    uint _S10 = as_type<uint>(kernelContext_4->payload_0[(base_0)>>2]);
    uint _S11 = as_type<uint>(kernelContext_4->payload_0[(base_0 + 4U)>>2]);
    uint _S12 = as_type<uint>(kernelContext_4->payload_0[(base_0 + 8U)>>2]);
    uint _S13 = as_type<uint>(kernelContext_4->payload_0[(base_0 + 12U)>>2]);
    uint _S14 = as_type<uint>(kernelContext_4->payload_0[(uint(int(base_0 + 16U)))>>2]);
    if(shift_2 == 0U)
    {
        (*w_0)[int(0)] = _S10;
        (*w_0)[int(1)] = _S11;
        (*w_0)[int(2)] = _S12;
        (*w_0)[int(3)] = _S13;
        return;
    }
    uint _S15 = 32U - shift_2;
    (*w_0)[int(0)] = (_S10 >> shift_2) | (_S11 << _S15);
    (*w_0)[int(1)] = (_S11 >> shift_2) | (_S12 << _S15);
    (*w_0)[int(2)] = (_S12 >> shift_2) | (_S13 << _S15);
    (*w_0)[int(3)] = (_S13 >> shift_2) | (_S14 << _S15);
    return;
}

uint body_word_0(const array<uint, int(8)> thread* row_0, uint c_0)
{
    return (((*row_0)[c_0 >> 2U]) >> ((c_0 & 3U) * 8U)) & 255U;
}

void pfor_lane_0(uint base_1, uint t_0, array<uint, int(8)> thread* v_0, KernelContext_0 thread* kernelContext_5)
{
    uint _S16 = load8_0(base_1, kernelContext_5);
    uint _S17 = load8_0(base_1 + 1U, kernelContext_5);
    thread array<uint, int(8)> bitmap_0;
    bitmap_0[0U] = 0U;
    bitmap_0[1U] = 0U;
    bitmap_0[2U] = 0U;
    bitmap_0[3U] = 0U;
    bitmap_0[4U] = 0U;
    bitmap_0[5U] = 0U;
    bitmap_0[6U] = 0U;
    bitmap_0[7U] = 0U;
    uint _S18 = base_1 + 2U;
    uint body_0;
    uint eb_0;
    uint excStart_0;
    uint rank_0;
    uint off_0;
    if(_S17 != 0U)
    {
        uint _S19 = load8_0(_S18, kernelContext_5);
        uint eb_1 = _S19 & 127U;
        uint posStart_0 = base_1 + 3U;
        if((_S19 >> 7U) == 0U)
        {
            if(t_0 < 8U)
            {
                uint _S20 = load32_0(posStart_0 + t_0 * 4U, kernelContext_5);
                body_0 = _S20;
            }
            else
            {
                body_0 = 0U;
            }
            uint _S21 = (simd_shuffle((body_0), ushort((int(0)))));
            bitmap_0[0U] = _S21;
            uint _S22 = (simd_shuffle((body_0), ushort((int(1)))));
            bitmap_0[1U] = _S22;
            uint _S23 = (simd_shuffle((body_0), ushort((int(2)))));
            bitmap_0[2U] = _S23;
            uint _S24 = (simd_shuffle((body_0), ushort((int(3)))));
            bitmap_0[3U] = _S24;
            uint _S25 = (simd_shuffle((body_0), ushort((int(4)))));
            bitmap_0[4U] = _S25;
            uint _S26 = (simd_shuffle((body_0), ushort((int(5)))));
            bitmap_0[5U] = _S26;
            uint _S27 = (simd_shuffle((body_0), ushort((int(6)))));
            bitmap_0[6U] = _S27;
            uint _S28 = (simd_shuffle((body_0), ushort((int(7)))));
            bitmap_0[7U] = _S28;
            body_0 = posStart_0 + 32U;
        }
        else
        {
            uint _S29 = (_S17 + 32U - 1U) / 32U;
            body_0 = 0U;
            for(;;)
            {
                if(body_0 < _S29)
                {
                }
                else
                {
                    break;
                }
                uint i_0 = body_0 * 32U + t_0;
                if(i_0 < _S17)
                {
                    uint _S30 = load8_0(posStart_0 + i_0, kernelContext_5);
                    eb_0 = _S30;
                }
                else
                {
                    eb_0 = 65535U;
                }
                uint _S31 = eb_0 >> 5U;
                if(_S31 == 0U)
                {
                    excStart_0 = 1U << (eb_0 & 31U);
                }
                else
                {
                    excStart_0 = 0U;
                }
                uint _S32 = simd_or(excStart_0);
                bitmap_0[0U] = (bitmap_0[0U]) | _S32;
                if(_S31 == 1U)
                {
                    rank_0 = 1U << (eb_0 & 31U);
                }
                else
                {
                    rank_0 = 0U;
                }
                uint _S33 = simd_or(rank_0);
                bitmap_0[1U] = (bitmap_0[1U]) | _S33;
                if(_S31 == 2U)
                {
                    off_0 = 1U << (eb_0 & 31U);
                }
                else
                {
                    off_0 = 0U;
                }
                uint _S34 = simd_or(off_0);
                bitmap_0[2U] = (bitmap_0[2U]) | _S34;
                uint _S35;
                if(_S31 == 3U)
                {
                    _S35 = 1U << (eb_0 & 31U);
                }
                else
                {
                    _S35 = 0U;
                }
                uint _S36 = simd_or(_S35);
                bitmap_0[3U] = (bitmap_0[3U]) | _S36;
                uint _S37;
                if(_S31 == 4U)
                {
                    _S37 = 1U << (eb_0 & 31U);
                }
                else
                {
                    _S37 = 0U;
                }
                uint _S38 = simd_or(_S37);
                bitmap_0[4U] = (bitmap_0[4U]) | _S38;
                uint _S39;
                if(_S31 == 5U)
                {
                    _S39 = 1U << (eb_0 & 31U);
                }
                else
                {
                    _S39 = 0U;
                }
                uint _S40 = simd_or(_S39);
                bitmap_0[5U] = (bitmap_0[5U]) | _S40;
                uint _S41;
                if(_S31 == 6U)
                {
                    _S41 = 1U << (eb_0 & 31U);
                }
                else
                {
                    _S41 = 0U;
                }
                uint _S42 = simd_or(_S41);
                bitmap_0[6U] = (bitmap_0[6U]) | _S42;
                uint _S43;
                if(_S31 == 7U)
                {
                    _S43 = 1U << (eb_0 & 31U);
                }
                else
                {
                    _S43 = 0U;
                }
                uint _S44 = simd_or(_S43);
                bitmap_0[7U] = (bitmap_0[7U]) | _S44;
                body_0 = body_0 + 1U;
            }
            body_0 = posStart_0 + _S17;
        }
        uint _S45 = body_0;
        body_0 = body_0 + ((_S17 * eb_1 + 7U) >> 3U);
        eb_0 = eb_1;
        excStart_0 = _S45;
    }
    else
    {
        body_0 = _S18;
        eb_0 = 0U;
        excStart_0 = _S18;
    }
    uint i0_0 = t_0 * 8U;
    uint _S46 = t_0 >> 2U;
    uint _S47 = (t_0 & 3U) * 8U;
    uint _S48 = ((bitmap_0[_S46]) >> _S47) & 255U;
    if(0U < _S46)
    {
        rank_0 = ((uint)popcount((bitmap_0[0U])));
    }
    else
    {
        rank_0 = 0U;
    }
    if(1U < _S46)
    {
        rank_0 = rank_0 + ((uint)popcount((bitmap_0[1U])));
    }
    if(2U < _S46)
    {
        rank_0 = rank_0 + ((uint)popcount((bitmap_0[2U])));
    }
    if(3U < _S46)
    {
        rank_0 = rank_0 + ((uint)popcount((bitmap_0[3U])));
    }
    if(4U < _S46)
    {
        rank_0 = rank_0 + ((uint)popcount((bitmap_0[4U])));
    }
    if(5U < _S46)
    {
        rank_0 = rank_0 + ((uint)popcount((bitmap_0[5U])));
    }
    if(6U < _S46)
    {
        rank_0 = rank_0 + ((uint)popcount((bitmap_0[6U])));
    }
    if(7U < _S46)
    {
        rank_0 = rank_0 + ((uint)popcount((bitmap_0[7U])));
    }
    uint rank_1 = rank_0 + ((uint)popcount(((bitmap_0[_S46]) & (low_mask_0(_S47)))));
    thread array<uint, int(8)> row_1;
    row_1[0U] = 0U;
    row_1[1U] = 0U;
    row_1[2U] = 0U;
    row_1[3U] = 0U;
    row_1[4U] = 0U;
    row_1[5U] = 0U;
    row_1[6U] = 0U;
    row_1[7U] = 0U;
    if(_S16 != 0U)
    {
        uint bitpos_0 = i0_0 / 16U * _S16;
        uint off_1 = bitpos_0 % 8U;
        uint a_5 = body_0 + bitpos_0 / 8U * 16U + i0_0 % 16U;
        thread array<uint, int(4)> w_1;
        load128_0(a_5, &w_1, kernelContext_5);
        row_1[int(0)] = w_1[int(0)];
        row_1[int(1)] = w_1[int(1)];
        row_1[int(2)] = w_1[int(2)];
        row_1[int(3)] = w_1[int(3)];
        if((off_1 + _S16) > 8U)
        {
            load128_0(a_5 + 16U, &w_1, kernelContext_5);
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
    uint _S49 = low_mask_0(_S16);
    thread array<uint, int(8)> _S50 = row_1;
    uint _S51 = body_word_0(&_S50, 0U);
    thread array<uint, int(8)> _S52 = row_1;
    uint _S53 = body_word_0(&_S52, 16U);
    (*v_0)[0U] = ((_S51 | (_S53 << 8U)) >> off_0) & _S49;
    thread array<uint, int(8)> _S54 = row_1;
    uint _S55 = body_word_0(&_S54, 1U);
    thread array<uint, int(8)> _S56 = row_1;
    uint _S57 = body_word_0(&_S56, 17U);
    (*v_0)[1U] = ((_S55 | (_S57 << 8U)) >> off_0) & _S49;
    thread array<uint, int(8)> _S58 = row_1;
    uint _S59 = body_word_0(&_S58, 2U);
    thread array<uint, int(8)> _S60 = row_1;
    uint _S61 = body_word_0(&_S60, 18U);
    (*v_0)[2U] = ((_S59 | (_S61 << 8U)) >> off_0) & _S49;
    thread array<uint, int(8)> _S62 = row_1;
    uint _S63 = body_word_0(&_S62, 3U);
    thread array<uint, int(8)> _S64 = row_1;
    uint _S65 = body_word_0(&_S64, 19U);
    (*v_0)[3U] = ((_S63 | (_S65 << 8U)) >> off_0) & _S49;
    thread array<uint, int(8)> _S66 = row_1;
    uint _S67 = body_word_0(&_S66, 4U);
    thread array<uint, int(8)> _S68 = row_1;
    uint _S69 = body_word_0(&_S68, 20U);
    (*v_0)[4U] = ((_S67 | (_S69 << 8U)) >> off_0) & _S49;
    thread array<uint, int(8)> _S70 = row_1;
    uint _S71 = body_word_0(&_S70, 5U);
    thread array<uint, int(8)> _S72 = row_1;
    uint _S73 = body_word_0(&_S72, 21U);
    (*v_0)[5U] = ((_S71 | (_S73 << 8U)) >> off_0) & _S49;
    thread array<uint, int(8)> _S74 = row_1;
    uint _S75 = body_word_0(&_S74, 6U);
    thread array<uint, int(8)> _S76 = row_1;
    uint _S77 = body_word_0(&_S76, 22U);
    (*v_0)[6U] = ((_S75 | (_S77 << 8U)) >> off_0) & _S49;
    thread array<uint, int(8)> _S78 = row_1;
    uint _S79 = body_word_0(&_S78, 7U);
    thread array<uint, int(8)> _S80 = row_1;
    uint _S81 = body_word_0(&_S80, 23U);
    (*v_0)[7U] = ((_S79 | (_S81 << 8U)) >> off_0) & _S49;
    if(((_S48 >> 0U) & 1U) != 0U)
    {
        uint bitpos_1 = (rank_1 + ((uint)popcount((_S48 & (low_mask_0(0U)))))) * eb_0;
        uint _S82 = load32_0(excStart_0 + (bitpos_1 >> 3U), kernelContext_5);
        (*v_0)[0U] = ((*v_0)[0U]) | (((_S82 >> (bitpos_1 & 7U)) & (low_mask_0(eb_0))) << _S16);
    }
    if(((_S48 >> 1U) & 1U) != 0U)
    {
        uint bitpos_2 = (rank_1 + ((uint)popcount((_S48 & (low_mask_0(1U)))))) * eb_0;
        uint _S83 = load32_0(excStart_0 + (bitpos_2 >> 3U), kernelContext_5);
        (*v_0)[1U] = ((*v_0)[1U]) | (((_S83 >> (bitpos_2 & 7U)) & (low_mask_0(eb_0))) << _S16);
    }
    if(((_S48 >> 2U) & 1U) != 0U)
    {
        uint bitpos_3 = (rank_1 + ((uint)popcount((_S48 & (low_mask_0(2U)))))) * eb_0;
        uint _S84 = load32_0(excStart_0 + (bitpos_3 >> 3U), kernelContext_5);
        (*v_0)[2U] = ((*v_0)[2U]) | (((_S84 >> (bitpos_3 & 7U)) & (low_mask_0(eb_0))) << _S16);
    }
    if(((_S48 >> 3U) & 1U) != 0U)
    {
        uint bitpos_4 = (rank_1 + ((uint)popcount((_S48 & (low_mask_0(3U)))))) * eb_0;
        uint _S85 = load32_0(excStart_0 + (bitpos_4 >> 3U), kernelContext_5);
        (*v_0)[3U] = ((*v_0)[3U]) | (((_S85 >> (bitpos_4 & 7U)) & (low_mask_0(eb_0))) << _S16);
    }
    if(((_S48 >> 4U) & 1U) != 0U)
    {
        uint bitpos_5 = (rank_1 + ((uint)popcount((_S48 & (low_mask_0(4U)))))) * eb_0;
        uint _S86 = load32_0(excStart_0 + (bitpos_5 >> 3U), kernelContext_5);
        (*v_0)[4U] = ((*v_0)[4U]) | (((_S86 >> (bitpos_5 & 7U)) & (low_mask_0(eb_0))) << _S16);
    }
    if(((_S48 >> 5U) & 1U) != 0U)
    {
        uint bitpos_6 = (rank_1 + ((uint)popcount((_S48 & (low_mask_0(5U)))))) * eb_0;
        uint _S87 = load32_0(excStart_0 + (bitpos_6 >> 3U), kernelContext_5);
        (*v_0)[5U] = ((*v_0)[5U]) | (((_S87 >> (bitpos_6 & 7U)) & (low_mask_0(eb_0))) << _S16);
    }
    if(((_S48 >> 6U) & 1U) != 0U)
    {
        uint bitpos_7 = (rank_1 + ((uint)popcount((_S48 & (low_mask_0(6U)))))) * eb_0;
        uint _S88 = load32_0(excStart_0 + (bitpos_7 >> 3U), kernelContext_5);
        (*v_0)[6U] = ((*v_0)[6U]) | (((_S88 >> (bitpos_7 & 7U)) & (low_mask_0(eb_0))) << _S16);
    }
    if(((_S48 >> 7U) & 1U) != 0U)
    {
        uint bitpos_8 = (rank_1 + ((uint)popcount((_S48 & (low_mask_0(7U)))))) * eb_0;
        uint _S89 = load32_0(excStart_0 + (bitpos_8 >> 3U), kernelContext_5);
        (*v_0)[7U] = ((*v_0)[7U]) | (((_S89 >> (bitpos_8 & 7U)) & (low_mask_0(eb_0))) << _S16);
    }
    return;
}

bool aligned_run_0(uint elt_0)
{
    return (elt_0 & 7U) == 0U;
}

void store_run_0(uint elt_1, const array<uint, int(8)> thread* v_1, KernelContext_0 thread* kernelContext_6)
{
    uint _S90 = (((((*v_1)[int(4)]) & 255U) | ((((*v_1)[int(5)]) & 255U) << 8U)) | ((((*v_1)[int(6)]) & 255U) << 16U)) | (((*v_1)[int(7)]) << 24U);
    kernelContext_6->outputBytes_0[(elt_1)>>2] = as_type<uint32_t>((((((*v_1)[int(0)]) & 255U) | ((((*v_1)[int(1)]) & 255U) << 8U)) | ((((*v_1)[int(2)]) & 255U) << 16U)) | (((*v_1)[int(3)]) << 24U));
    kernelContext_6->outputBytes_0[(elt_1 + 4U)>>2] = as_type<uint32_t>(_S90);
    return;
}

[[kernel]] void unpack_blocks(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_0 [[thread_position_in_threadgroup]], block_task_0 device* blocks_1 [[buffer(1)]], uint32_t device* payload_1 [[buffer(0)]], uint32_t device* outputBytes_1 [[buffer(3)]], uchar device* output_1 [[buffer(2)]], block_chunk_0 device* blockChunks_1 [[buffer(8)]], band_frame_0 device* bandFrames_1 [[buffer(4)]], uint device* chunkSizes_1 [[buffer(9)]], block_ref_0 device* bandBlocks_1 [[buffer(6)]], uint device* status_1 [[buffer(7)]], strip_task_0 device* strips_1 [[buffer(5)]], raw_copy_0 device* rawCopies_1 [[buffer(11)]], uchar device* rawValues_1 [[buffer(10)]])
{
    thread KernelContext_0 kernelContext_7;
    (&kernelContext_7)->blocks_0 = blocks_1;
    (&kernelContext_7)->payload_0 = payload_1;
    (&kernelContext_7)->outputBytes_0 = outputBytes_1;
    (&kernelContext_7)->output_0 = output_1;
    (&kernelContext_7)->blockChunks_0 = blockChunks_1;
    (&kernelContext_7)->bandFrames_0 = bandFrames_1;
    (&kernelContext_7)->chunkSizes_0 = chunkSizes_1;
    (&kernelContext_7)->bandBlocks_0 = bandBlocks_1;
    (&kernelContext_7)->status_0 = status_1;
    (&kernelContext_7)->strips_0 = strips_1;
    (&kernelContext_7)->rawCopies_0 = rawCopies_1;
    (&kernelContext_7)->rawValues_0 = rawValues_1;
    uint _S91 = thread_0.x;
    uint t_1 = _S91 % 32U;
    block_task_0 task_0 = blocks_1[group_0.x * 4U + _S91 / 32U];
    thread array<uint, int(8)> v_2;
    pfor_lane_0(task_0.payOff_0, t_1, &v_2, &kernelContext_7);
    uint i0_1 = t_1 * 8U;
    uint elt_2 = task_0.outBase_0 + i0_1;
    bool _S92;
    if((i0_1 + 8U) <= (task_0.count_0))
    {
        _S92 = aligned_run_0(elt_2);
    }
    else
    {
        _S92 = false;
    }
    if(_S92)
    {
        thread array<uint, int(8)> _S93 = v_2;
        store_run_0(elt_2, &_S93, &kernelContext_7);
    }
    else
    {
        if(i0_1 < (task_0.count_0))
        {
            *((&kernelContext_7)->output_0+elt_2) = uchar(v_2[0U]);
        }
        if((i0_1 + 1U) < (task_0.count_0))
        {
            *((&kernelContext_7)->output_0+(elt_2 + 1U)) = uchar(v_2[1U]);
        }
        if((i0_1 + 2U) < (task_0.count_0))
        {
            *((&kernelContext_7)->output_0+(elt_2 + 2U)) = uchar(v_2[2U]);
        }
        if((i0_1 + 3U) < (task_0.count_0))
        {
            *((&kernelContext_7)->output_0+(elt_2 + 3U)) = uchar(v_2[3U]);
        }
        if((i0_1 + 4U) < (task_0.count_0))
        {
            *((&kernelContext_7)->output_0+(elt_2 + 4U)) = uchar(v_2[4U]);
        }
        if((i0_1 + 5U) < (task_0.count_0))
        {
            *((&kernelContext_7)->output_0+(elt_2 + 5U)) = uchar(v_2[5U]);
        }
        if((i0_1 + 6U) < (task_0.count_0))
        {
            *((&kernelContext_7)->output_0+(elt_2 + 6U)) = uchar(v_2[6U]);
        }
        if((i0_1 + 7U) < (task_0.count_0))
        {
            *((&kernelContext_7)->output_0+(elt_2 + 7U)) = uchar(v_2[7U]);
        }
    }
    return;
}

struct frame_rest_0
{
    bool bad_0;
    uint bytes_0;
    uint cap_0;
};

frame_rest_0 rest_of_0(const band_frame_0 thread* f_0)
{
    thread frame_rest_0 r_0;
    uint _S94 = f_0->blockCount_0;
    uint _S95 = f_0->paySize_0;
    bool _S96 = (f_0->blockCount_0) > (f_0->paySize_0 / 3U);
    (&r_0)->bad_0 = _S96;
    uint _S97;
    if(_S96)
    {
        _S97 = 0U;
    }
    else
    {
        _S97 = _S95 - 3U * _S94;
    }
    (&r_0)->bytes_0 = _S97;
    (&r_0)->cap_0 = _S97 + 1U;
    return r_0;
}

uint block_size_0(const band_frame_0 thread* f_1, uint k_0, bool thread* bad_1, KernelContext_0 thread* kernelContext_8)
{
    uint at_0 = f_1->payOff_1 + 3U * k_0;
    uint _S98 = load8_0(at_0, kernelContext_8);
    uint _S99 = load8_0(at_0 + 1U, kernelContext_8);
    uint _S100 = load8_0(at_0 + 2U, kernelContext_8);
    uint size_0 = 32U * _S98;
    if(_S98 > 8U)
    {
        *bad_1 = true;
    }
    uint size_1;
    if(_S99 != 0U)
    {
        uint eb_2 = _S100 & 127U;
        bool _S101;
        if(eb_2 == 0U)
        {
            _S101 = true;
        }
        else
        {
            _S101 = (_S98 + eb_2) > 8U;
        }
        if(_S101)
        {
            *bad_1 = true;
        }
        if((_S100 >> 7U) != 0U)
        {
            size_1 = _S99;
        }
        else
        {
            size_1 = 32U;
        }
        size_1 = size_0 + (size_1 + ((_S99 * eb_2 + 7U) >> 3U));
    }
    else
    {
        if(_S100 != 0U)
        {
            *bad_1 = true;
        }
        size_1 = size_0;
    }
    if(*bad_1)
    {
        size_1 = 0U;
    }
    return size_1;
}

void size_chunk_0(uint index_0, uint t_2, KernelContext_0 thread* kernelContext_9)
{
    block_chunk_0 c_1 = kernelContext_9->blockChunks_0[index_0];
    if((c_1.count_1) == 0U)
    {
        return;
    }
    band_frame_0 f_2 = kernelContext_9->bandFrames_0[c_1.frame_0];
    thread band_frame_0 _S102 = f_2;
    frame_rest_0 _S103 = rest_of_0(&_S102);
    uint total_0;
    if(!_S103.bad_0)
    {
        total_0 = 0U;
        uint k0_0 = 0U;
        for(;;)
        {
            if(k0_0 < (c_1.count_1))
            {
            }
            else
            {
                break;
            }
            uint k_1 = k0_0 + t_2;
            thread bool bad_2 = false;
            uint size_2;
            if(k_1 < (c_1.count_1))
            {
                uint _S104 = c_1.first_0 + k_1;
                thread band_frame_0 _S105 = f_2;
                uint _S106 = block_size_0(&_S105, _S104, &bad_2, kernelContext_9);
                size_2 = _S106;
            }
            else
            {
                size_2 = 0U;
            }
            uint _S107 = simd_sum(size_2);
            uint _S108 = min(total_0 + _S107, _S103.cap_0);
            uint k0_1 = k0_0 + 32U;
            total_0 = _S108;
            k0_0 = k0_1;
        }
    }
    else
    {
        total_0 = 0U;
    }
    if(t_2 == 0U)
    {
        *(kernelContext_9->chunkSizes_0+index_0) = total_0;
    }
    return;
}

[[kernel]] void size_chunks(uint3 group_1 [[threadgroup_position_in_grid]], uint3 thread_1 [[thread_position_in_threadgroup]], block_task_0 device* blocks_2 [[buffer(1)]], uint32_t device* payload_2 [[buffer(0)]], uint32_t device* outputBytes_2 [[buffer(3)]], uchar device* output_2 [[buffer(2)]], block_chunk_0 device* blockChunks_2 [[buffer(8)]], band_frame_0 device* bandFrames_2 [[buffer(4)]], uint device* chunkSizes_2 [[buffer(9)]], block_ref_0 device* bandBlocks_2 [[buffer(6)]], uint device* status_2 [[buffer(7)]], strip_task_0 device* strips_2 [[buffer(5)]], raw_copy_0 device* rawCopies_2 [[buffer(11)]], uchar device* rawValues_2 [[buffer(10)]])
{
    thread KernelContext_0 kernelContext_10;
    (&kernelContext_10)->blocks_0 = blocks_2;
    (&kernelContext_10)->payload_0 = payload_2;
    (&kernelContext_10)->outputBytes_0 = outputBytes_2;
    (&kernelContext_10)->output_0 = output_2;
    (&kernelContext_10)->blockChunks_0 = blockChunks_2;
    (&kernelContext_10)->bandFrames_0 = bandFrames_2;
    (&kernelContext_10)->chunkSizes_0 = chunkSizes_2;
    (&kernelContext_10)->bandBlocks_0 = bandBlocks_2;
    (&kernelContext_10)->status_0 = status_2;
    (&kernelContext_10)->strips_0 = strips_2;
    (&kernelContext_10)->rawCopies_0 = rawCopies_2;
    (&kernelContext_10)->rawValues_0 = rawValues_2;
    uint _S109 = thread_1.x;
    size_chunk_0(group_1.x * 4U + _S109 / 32U, _S109 % 32U, &kernelContext_10);
    return;
}

uint sat_add_0(uint a_6, uint b_1)
{
    uint sum_0 = a_6 + b_1;
    uint _S110;
    if(sum_0 < a_6)
    {
        _S110 = 4294967295U;
    }
    else
    {
        _S110 = sum_0;
    }
    return _S110;
}

void report_0(uint check_0, uint frame_3, KernelContext_0 thread* kernelContext_11)
{
    uint _S111 = atomic_fetch_min_explicit(((atomic_uint device*)(kernelContext_11->status_0+check_0)), frame_3, memory_order_relaxed);
    return;
}

void plan_chunk_0(uint index_1, uint t_3, KernelContext_0 thread* kernelContext_12)
{
    block_chunk_0 c_2 = kernelContext_12->blockChunks_0[index_1];
    if((c_2.count_1) == 0U)
    {
        return;
    }
    band_frame_0 f_3 = kernelContext_12->bandFrames_0[c_2.frame_0];
    thread band_frame_0 _S112 = f_3;
    frame_rest_0 _S113 = rest_of_0(&_S112);
    uint _S114 = f_3.payOff_1 + 3U * f_3.blockCount_0;
    uint j_0 = c_2.firstChunk_0 + t_3;
    uint carry_0 = 0U;
    for(;;)
    {
        if(j_0 < index_1)
        {
        }
        else
        {
            break;
        }
        uint _S115 = min(sat_add_0(carry_0, *(kernelContext_12->chunkSizes_0+j_0)), _S113.cap_0);
        j_0 = j_0 + 32U;
        carry_0 = _S115;
    }
    uint _S116 = (simd_shuffle_xor((carry_0), ushort((1U))));
    uint _S117 = min(sat_add_0(carry_0, _S116), _S113.cap_0);
    uint _S118 = (simd_shuffle_xor((_S117), ushort((2U))));
    uint _S119 = min(sat_add_0(_S117, _S118), _S113.cap_0);
    uint _S120 = (simd_shuffle_xor((_S119), ushort((4U))));
    uint _S121 = min(sat_add_0(_S119, _S120), _S113.cap_0);
    uint _S122 = (simd_shuffle_xor((_S121), ushort((8U))));
    uint _S123 = min(sat_add_0(_S121, _S122), _S113.cap_0);
    uint _S124 = (simd_shuffle_xor((_S123), ushort((16U))));
    uint _S125 = min(sat_add_0(_S123, _S124), _S113.cap_0);
    uint k0_2 = 0U;
    carry_0 = _S125;
    bool anyBad_0 = false;
    for(;;)
    {
        if(k0_2 < (c_2.count_1))
        {
        }
        else
        {
            break;
        }
        uint k_2 = k0_2 + t_3;
        thread bool bad_3 = _S113.bad_0;
        bool _S126 = k_2 < (c_2.count_1);
        bool _S127;
        if(_S126)
        {
            _S127 = !_S113.bad_0;
        }
        else
        {
            _S127 = false;
        }
        uint size_3;
        if(_S127)
        {
            uint _S128 = c_2.first_0 + k_2;
            thread band_frame_0 _S129 = f_3;
            uint _S130 = block_size_0(&_S129, _S128, &bad_3, kernelContext_12);
            size_3 = _S130;
        }
        else
        {
            size_3 = 0U;
        }
        uint _S131 = simd_prefix_exclusive_sum(size_3);
        uint start_0 = carry_0 + _S131;
        if((start_0 + size_3) > (_S113.bytes_0))
        {
            bad_3 = true;
        }
        if(_S126)
        {
            thread block_ref_0 ref_0;
            (&ref_0)->hdr_0 = f_3.payOff_1 + 3U * (c_2.first_0 + k_2);
            if(bad_3)
            {
                j_0 = 4294967295U;
            }
            else
            {
                j_0 = _S114 + start_0;
            }
            (&ref_0)->rest_0 = j_0;
            *(kernelContext_12->bandBlocks_0+(f_3.firstBlock_0 + c_2.first_0 + k_2)) = ref_0;
        }
        if(anyBad_0)
        {
            anyBad_0 = true;
        }
        else
        {
            bool _S132;
            if(bad_3)
            {
                _S132 = _S126;
            }
            else
            {
                _S132 = false;
            }
            bool _S133 = simd_any(_S132);
            anyBad_0 = _S133;
        }
        uint _S134 = simd_sum(size_3);
        uint _S135 = min(carry_0 + _S134, _S113.cap_0);
        k0_2 = k0_2 + 32U;
        carry_0 = _S135;
    }
    bool last_0 = (c_2.first_0 + c_2.count_1) == (f_3.blockCount_0);
    if(anyBad_0)
    {
        anyBad_0 = true;
    }
    else
    {
        anyBad_0 = _S113.bad_0;
    }
    if(anyBad_0)
    {
        anyBad_0 = true;
    }
    else
    {
        if(last_0)
        {
            anyBad_0 = carry_0 != (_S113.bytes_0);
        }
        else
        {
            anyBad_0 = false;
        }
    }
    if(anyBad_0)
    {
        anyBad_0 = t_3 == 0U;
    }
    else
    {
        anyBad_0 = false;
    }
    if(anyBad_0)
    {
        report_0(0U, f_3.frame_1, kernelContext_12);
    }
    return;
}

[[kernel]] void plan_blocks(uint3 group_2 [[threadgroup_position_in_grid]], uint3 thread_2 [[thread_position_in_threadgroup]], block_task_0 device* blocks_3 [[buffer(1)]], uint32_t device* payload_3 [[buffer(0)]], uint32_t device* outputBytes_3 [[buffer(3)]], uchar device* output_3 [[buffer(2)]], block_chunk_0 device* blockChunks_3 [[buffer(8)]], band_frame_0 device* bandFrames_3 [[buffer(4)]], uint device* chunkSizes_3 [[buffer(9)]], block_ref_0 device* bandBlocks_3 [[buffer(6)]], uint device* status_3 [[buffer(7)]], strip_task_0 device* strips_3 [[buffer(5)]], raw_copy_0 device* rawCopies_3 [[buffer(11)]], uchar device* rawValues_3 [[buffer(10)]])
{
    thread KernelContext_0 kernelContext_13;
    (&kernelContext_13)->blocks_0 = blocks_3;
    (&kernelContext_13)->payload_0 = payload_3;
    (&kernelContext_13)->outputBytes_0 = outputBytes_3;
    (&kernelContext_13)->output_0 = output_3;
    (&kernelContext_13)->blockChunks_0 = blockChunks_3;
    (&kernelContext_13)->bandFrames_0 = bandFrames_3;
    (&kernelContext_13)->chunkSizes_0 = chunkSizes_3;
    (&kernelContext_13)->bandBlocks_0 = bandBlocks_3;
    (&kernelContext_13)->status_0 = status_3;
    (&kernelContext_13)->strips_0 = strips_3;
    (&kernelContext_13)->rawCopies_0 = rawCopies_3;
    (&kernelContext_13)->rawValues_0 = rawValues_3;
    uint _S136 = thread_2.x;
    plan_chunk_0(group_2.x * 4U + _S136 / 32U, _S136 % 32U, &kernelContext_13);
    return;
}

void block_exceptions_0(uint rest_1, uint nexc_0, bool list_0, uint t_4, uint thread* flags_0, uint thread* rank_2, bool thread* bad_4, KernelContext_0 thread* kernelContext_14)
{
    *flags_0 = 0U;
    *rank_2 = 0U;
    uint _S137 = t_4 >> 2U;
    uint shift_3 = (t_4 & 3U) * 8U;
    uint chunk_0;
    uint total_1;
    uint mine_0;
    if(!list_0)
    {
        if(t_4 < 8U)
        {
            uint _S138 = load32_1(rest_1 + t_4 * 4U, kernelContext_14);
            chunk_0 = _S138;
        }
        else
        {
            chunk_0 = 0U;
        }
        uint m_0 = (simd_broadcast((chunk_0), ushort((0U))));
        uint n_0 = ((uint)popcount((m_0)));
        if(0U < _S137)
        {
            *rank_2 = *rank_2 + n_0;
        }
        if(0U == _S137)
        {
            mine_0 = m_0;
        }
        else
        {
            mine_0 = 0U;
        }
        uint m_1 = (simd_broadcast((chunk_0), ushort((1U))));
        uint n_1 = ((uint)popcount((m_1)));
        uint total_2 = n_0 + n_1;
        if(1U < _S137)
        {
            *rank_2 = *rank_2 + n_1;
        }
        if(1U == _S137)
        {
            mine_0 = m_1;
        }
        uint m_2 = (simd_broadcast((chunk_0), ushort((2U))));
        uint n_2 = ((uint)popcount((m_2)));
        uint total_3 = total_2 + n_2;
        if(2U < _S137)
        {
            *rank_2 = *rank_2 + n_2;
        }
        if(2U == _S137)
        {
            mine_0 = m_2;
        }
        uint m_3 = (simd_broadcast((chunk_0), ushort((3U))));
        uint n_3 = ((uint)popcount((m_3)));
        uint total_4 = total_3 + n_3;
        if(3U < _S137)
        {
            *rank_2 = *rank_2 + n_3;
        }
        if(3U == _S137)
        {
            mine_0 = m_3;
        }
        uint m_4 = (simd_broadcast((chunk_0), ushort((4U))));
        uint n_4 = ((uint)popcount((m_4)));
        uint total_5 = total_4 + n_4;
        if(4U < _S137)
        {
            *rank_2 = *rank_2 + n_4;
        }
        if(4U == _S137)
        {
            mine_0 = m_4;
        }
        uint m_5 = (simd_broadcast((chunk_0), ushort((5U))));
        uint n_5 = ((uint)popcount((m_5)));
        uint total_6 = total_5 + n_5;
        if(5U < _S137)
        {
            *rank_2 = *rank_2 + n_5;
        }
        if(5U == _S137)
        {
            mine_0 = m_5;
        }
        uint m_6 = (simd_broadcast((chunk_0), ushort((6U))));
        uint n_6 = ((uint)popcount((m_6)));
        uint total_7 = total_6 + n_6;
        if(6U < _S137)
        {
            *rank_2 = *rank_2 + n_6;
        }
        if(6U == _S137)
        {
            mine_0 = m_6;
        }
        uint m_7 = (simd_broadcast((chunk_0), ushort((7U))));
        uint n_7 = ((uint)popcount((m_7)));
        uint total_8 = total_7 + n_7;
        if(7U < _S137)
        {
            *rank_2 = *rank_2 + n_7;
        }
        if(7U == _S137)
        {
            mine_0 = m_7;
        }
        total_1 = total_8;
    }
    else
    {
        chunk_0 = 0U;
        uint last_1 = 0U;
        total_1 = 0U;
        mine_0 = 0U;
        for(;;)
        {
            uint _S139 = chunk_0 * 32U;
            if(_S139 < nexc_0)
            {
            }
            else
            {
                break;
            }
            uint i_1 = _S139 + t_4;
            bool _S140 = i_1 < nexc_0;
            uint p_0;
            if(_S140)
            {
                uint _S141 = load8_1(rest_1 + i_1, kernelContext_14);
                p_0 = _S141;
            }
            else
            {
                p_0 = 65535U;
            }
            uint _S142 = (simd_shuffle_up((p_0), ushort((1U))));
            uint prev_0;
            if(t_4 == 0U)
            {
                prev_0 = last_1;
            }
            else
            {
                prev_0 = _S142;
            }
            bool _S143;
            if(_S140)
            {
                _S143 = i_1 != 0U;
            }
            else
            {
                _S143 = false;
            }
            bool _S144;
            if(_S143)
            {
                _S144 = prev_0 >= p_0;
            }
            else
            {
                _S144 = false;
            }
            if(_S144)
            {
                *bad_4 = true;
            }
            uint _S145 = (simd_broadcast((p_0), ushort((31U))));
            uint _S146 = p_0 >> 5U;
            uint _S147;
            if(_S146 == 0U)
            {
                _S147 = 1U << (p_0 & 31U);
            }
            else
            {
                _S147 = 0U;
            }
            uint m_8 = (simd_or((_S147)));
            uint n_8 = ((uint)popcount((m_8)));
            uint total_9 = total_1 + n_8;
            if(0U < _S137)
            {
                *rank_2 = *rank_2 + n_8;
            }
            uint mine_1;
            if(0U == _S137)
            {
                mine_1 = mine_0 | m_8;
            }
            else
            {
                mine_1 = mine_0;
            }
            uint _S148;
            if(_S146 == 1U)
            {
                _S148 = 1U << (p_0 & 31U);
            }
            else
            {
                _S148 = 0U;
            }
            uint m_9 = (simd_or((_S148)));
            uint n_9 = ((uint)popcount((m_9)));
            uint total_10 = total_9 + n_9;
            if(1U < _S137)
            {
                *rank_2 = *rank_2 + n_9;
            }
            uint mine_2;
            if(1U == _S137)
            {
                mine_2 = mine_1 | m_9;
            }
            else
            {
                mine_2 = mine_1;
            }
            uint _S149;
            if(_S146 == 2U)
            {
                _S149 = 1U << (p_0 & 31U);
            }
            else
            {
                _S149 = 0U;
            }
            uint m_10 = (simd_or((_S149)));
            uint n_10 = ((uint)popcount((m_10)));
            uint total_11 = total_10 + n_10;
            if(2U < _S137)
            {
                *rank_2 = *rank_2 + n_10;
            }
            uint mine_3;
            if(2U == _S137)
            {
                mine_3 = mine_2 | m_10;
            }
            else
            {
                mine_3 = mine_2;
            }
            uint _S150;
            if(_S146 == 3U)
            {
                _S150 = 1U << (p_0 & 31U);
            }
            else
            {
                _S150 = 0U;
            }
            uint m_11 = (simd_or((_S150)));
            uint n_11 = ((uint)popcount((m_11)));
            uint total_12 = total_11 + n_11;
            if(3U < _S137)
            {
                *rank_2 = *rank_2 + n_11;
            }
            uint mine_4;
            if(3U == _S137)
            {
                mine_4 = mine_3 | m_11;
            }
            else
            {
                mine_4 = mine_3;
            }
            uint _S151;
            if(_S146 == 4U)
            {
                _S151 = 1U << (p_0 & 31U);
            }
            else
            {
                _S151 = 0U;
            }
            uint m_12 = (simd_or((_S151)));
            uint n_12 = ((uint)popcount((m_12)));
            uint total_13 = total_12 + n_12;
            if(4U < _S137)
            {
                *rank_2 = *rank_2 + n_12;
            }
            uint mine_5;
            if(4U == _S137)
            {
                mine_5 = mine_4 | m_12;
            }
            else
            {
                mine_5 = mine_4;
            }
            uint _S152;
            if(_S146 == 5U)
            {
                _S152 = 1U << (p_0 & 31U);
            }
            else
            {
                _S152 = 0U;
            }
            uint m_13 = (simd_or((_S152)));
            uint n_13 = ((uint)popcount((m_13)));
            uint total_14 = total_13 + n_13;
            if(5U < _S137)
            {
                *rank_2 = *rank_2 + n_13;
            }
            uint mine_6;
            if(5U == _S137)
            {
                mine_6 = mine_5 | m_13;
            }
            else
            {
                mine_6 = mine_5;
            }
            uint _S153;
            if(_S146 == 6U)
            {
                _S153 = 1U << (p_0 & 31U);
            }
            else
            {
                _S153 = 0U;
            }
            uint m_14 = (simd_or((_S153)));
            uint n_14 = ((uint)popcount((m_14)));
            uint total_15 = total_14 + n_14;
            if(6U < _S137)
            {
                *rank_2 = *rank_2 + n_14;
            }
            uint mine_7;
            if(6U == _S137)
            {
                mine_7 = mine_6 | m_14;
            }
            else
            {
                mine_7 = mine_6;
            }
            uint _S154;
            if(_S146 == 7U)
            {
                _S154 = 1U << (p_0 & 31U);
            }
            else
            {
                _S154 = 0U;
            }
            uint m_15 = (simd_or((_S154)));
            uint n_15 = ((uint)popcount((m_15)));
            uint total_16 = total_15 + n_15;
            if(7U < _S137)
            {
                *rank_2 = *rank_2 + n_15;
            }
            if(7U == _S137)
            {
                mine_0 = mine_7 | m_15;
            }
            else
            {
                mine_0 = mine_7;
            }
            chunk_0 = chunk_0 + 1U;
            last_1 = _S145;
            total_1 = total_16;
        }
    }
    if(total_1 != nexc_0)
    {
        *bad_4 = true;
    }
    *flags_0 = (mine_0 >> shift_3) & 255U;
    *rank_2 = *rank_2 + ((uint)popcount((mine_0 & (low_mask_0(shift_3)))));
    return;
}

void block_lane_0(uint hdr_1, uint rest_2, uint t_5, array<uint, int(8)> thread* v_3, bool thread* bad_5, KernelContext_0 thread* kernelContext_15)
{
    uint _S155 = load8_0(hdr_1, kernelContext_15);
    uint _S156 = load8_0(hdr_1 + 1U, kernelContext_15);
    thread uint flags_1 = 0U;
    thread uint rank_3 = 0U;
    uint body_1;
    uint eb_3;
    uint excStart_1;
    if(_S156 != 0U)
    {
        uint _S157 = load8_0(hdr_1 + 2U, kernelContext_15);
        uint eb_4 = _S157 & 127U;
        bool list_1 = (_S157 >> 7U) != 0U;
        block_exceptions_0(rest_2, _S156, list_1, t_5, &flags_1, &rank_3, bad_5, kernelContext_15);
        if(list_1)
        {
            body_1 = _S156;
        }
        else
        {
            body_1 = 32U;
        }
        uint excStart_2 = rest_2 + body_1;
        body_1 = excStart_2 + ((_S156 * eb_4 + 7U) >> 3U);
        eb_3 = eb_4;
        excStart_1 = excStart_2;
    }
    else
    {
        body_1 = rest_2;
        eb_3 = 0U;
        excStart_1 = rest_2;
    }
    uint i0_2 = t_5 * 8U;
    thread array<uint, int(8)> row_2;
    row_2[0U] = 0U;
    row_2[1U] = 0U;
    row_2[2U] = 0U;
    row_2[3U] = 0U;
    row_2[4U] = 0U;
    row_2[5U] = 0U;
    row_2[6U] = 0U;
    row_2[7U] = 0U;
    uint off_2;
    if(_S155 != 0U)
    {
        uint bitpos_9 = i0_2 / 16U * _S155;
        uint off_3 = bitpos_9 % 8U;
        uint a_7 = body_1 + bitpos_9 / 8U * 16U + i0_2 % 16U;
        thread array<uint, int(4)> w4_0;
        load128_0(a_7, &w4_0, kernelContext_15);
        row_2[int(0)] = w4_0[int(0)];
        row_2[int(1)] = w4_0[int(1)];
        row_2[int(2)] = w4_0[int(2)];
        row_2[int(3)] = w4_0[int(3)];
        if((off_3 + _S155) > 8U)
        {
            load128_0(a_7 + 16U, &w4_0, kernelContext_15);
            row_2[int(4)] = w4_0[int(0)];
            row_2[int(5)] = w4_0[int(1)];
            row_2[int(6)] = w4_0[int(2)];
            row_2[int(7)] = w4_0[int(3)];
        }
        off_2 = off_3;
    }
    else
    {
        off_2 = 0U;
    }
    uint _S158 = low_mask_0(_S155);
    thread array<uint, int(8)> _S159 = row_2;
    uint _S160 = body_word_0(&_S159, 0U);
    thread array<uint, int(8)> _S161 = row_2;
    uint _S162 = body_word_0(&_S161, 16U);
    (*v_3)[0U] = ((_S160 | (_S162 << 8U)) >> off_2) & _S158;
    thread array<uint, int(8)> _S163 = row_2;
    uint _S164 = body_word_0(&_S163, 1U);
    thread array<uint, int(8)> _S165 = row_2;
    uint _S166 = body_word_0(&_S165, 17U);
    (*v_3)[1U] = ((_S164 | (_S166 << 8U)) >> off_2) & _S158;
    thread array<uint, int(8)> _S167 = row_2;
    uint _S168 = body_word_0(&_S167, 2U);
    thread array<uint, int(8)> _S169 = row_2;
    uint _S170 = body_word_0(&_S169, 18U);
    (*v_3)[2U] = ((_S168 | (_S170 << 8U)) >> off_2) & _S158;
    thread array<uint, int(8)> _S171 = row_2;
    uint _S172 = body_word_0(&_S171, 3U);
    thread array<uint, int(8)> _S173 = row_2;
    uint _S174 = body_word_0(&_S173, 19U);
    (*v_3)[3U] = ((_S172 | (_S174 << 8U)) >> off_2) & _S158;
    thread array<uint, int(8)> _S175 = row_2;
    uint _S176 = body_word_0(&_S175, 4U);
    thread array<uint, int(8)> _S177 = row_2;
    uint _S178 = body_word_0(&_S177, 20U);
    (*v_3)[4U] = ((_S176 | (_S178 << 8U)) >> off_2) & _S158;
    thread array<uint, int(8)> _S179 = row_2;
    uint _S180 = body_word_0(&_S179, 5U);
    thread array<uint, int(8)> _S181 = row_2;
    uint _S182 = body_word_0(&_S181, 21U);
    (*v_3)[5U] = ((_S180 | (_S182 << 8U)) >> off_2) & _S158;
    thread array<uint, int(8)> _S183 = row_2;
    uint _S184 = body_word_0(&_S183, 6U);
    thread array<uint, int(8)> _S185 = row_2;
    uint _S186 = body_word_0(&_S185, 22U);
    (*v_3)[6U] = ((_S184 | (_S186 << 8U)) >> off_2) & _S158;
    thread array<uint, int(8)> _S187 = row_2;
    uint _S188 = body_word_0(&_S187, 7U);
    thread array<uint, int(8)> _S189 = row_2;
    uint _S190 = body_word_0(&_S189, 23U);
    (*v_3)[7U] = ((_S188 | (_S190 << 8U)) >> off_2) & _S158;
    if(flags_1 != 0U)
    {
        uint index_2 = rank_3 + ((uint)popcount((flags_1 & (low_mask_0(0U)))));
        bool _S191;
        if(((flags_1 >> 0U) & 1U) != 0U)
        {
            _S191 = index_2 < _S156;
        }
        else
        {
            _S191 = false;
        }
        if(_S191)
        {
            uint bitpos_10 = index_2 * eb_3;
            uint _S192 = load32_0(excStart_1 + (bitpos_10 >> 3U), kernelContext_15);
            (*v_3)[0U] = ((*v_3)[0U]) | (((_S192 >> (bitpos_10 & 7U)) & (low_mask_0(eb_3))) << _S155);
        }
        uint index_3 = rank_3 + ((uint)popcount((flags_1 & (low_mask_0(1U)))));
        if(((flags_1 >> 1U) & 1U) != 0U)
        {
            _S191 = index_3 < _S156;
        }
        else
        {
            _S191 = false;
        }
        if(_S191)
        {
            uint bitpos_11 = index_3 * eb_3;
            uint _S193 = load32_0(excStart_1 + (bitpos_11 >> 3U), kernelContext_15);
            (*v_3)[1U] = ((*v_3)[1U]) | (((_S193 >> (bitpos_11 & 7U)) & (low_mask_0(eb_3))) << _S155);
        }
        uint index_4 = rank_3 + ((uint)popcount((flags_1 & (low_mask_0(2U)))));
        if(((flags_1 >> 2U) & 1U) != 0U)
        {
            _S191 = index_4 < _S156;
        }
        else
        {
            _S191 = false;
        }
        if(_S191)
        {
            uint bitpos_12 = index_4 * eb_3;
            uint _S194 = load32_0(excStart_1 + (bitpos_12 >> 3U), kernelContext_15);
            (*v_3)[2U] = ((*v_3)[2U]) | (((_S194 >> (bitpos_12 & 7U)) & (low_mask_0(eb_3))) << _S155);
        }
        uint index_5 = rank_3 + ((uint)popcount((flags_1 & (low_mask_0(3U)))));
        if(((flags_1 >> 3U) & 1U) != 0U)
        {
            _S191 = index_5 < _S156;
        }
        else
        {
            _S191 = false;
        }
        if(_S191)
        {
            uint bitpos_13 = index_5 * eb_3;
            uint _S195 = load32_0(excStart_1 + (bitpos_13 >> 3U), kernelContext_15);
            (*v_3)[3U] = ((*v_3)[3U]) | (((_S195 >> (bitpos_13 & 7U)) & (low_mask_0(eb_3))) << _S155);
        }
        uint index_6 = rank_3 + ((uint)popcount((flags_1 & (low_mask_0(4U)))));
        if(((flags_1 >> 4U) & 1U) != 0U)
        {
            _S191 = index_6 < _S156;
        }
        else
        {
            _S191 = false;
        }
        if(_S191)
        {
            uint bitpos_14 = index_6 * eb_3;
            uint _S196 = load32_0(excStart_1 + (bitpos_14 >> 3U), kernelContext_15);
            (*v_3)[4U] = ((*v_3)[4U]) | (((_S196 >> (bitpos_14 & 7U)) & (low_mask_0(eb_3))) << _S155);
        }
        uint index_7 = rank_3 + ((uint)popcount((flags_1 & (low_mask_0(5U)))));
        if(((flags_1 >> 5U) & 1U) != 0U)
        {
            _S191 = index_7 < _S156;
        }
        else
        {
            _S191 = false;
        }
        if(_S191)
        {
            uint bitpos_15 = index_7 * eb_3;
            uint _S197 = load32_0(excStart_1 + (bitpos_15 >> 3U), kernelContext_15);
            (*v_3)[5U] = ((*v_3)[5U]) | (((_S197 >> (bitpos_15 & 7U)) & (low_mask_0(eb_3))) << _S155);
        }
        uint index_8 = rank_3 + ((uint)popcount((flags_1 & (low_mask_0(6U)))));
        if(((flags_1 >> 6U) & 1U) != 0U)
        {
            _S191 = index_8 < _S156;
        }
        else
        {
            _S191 = false;
        }
        if(_S191)
        {
            uint bitpos_16 = index_8 * eb_3;
            uint _S198 = load32_0(excStart_1 + (bitpos_16 >> 3U), kernelContext_15);
            (*v_3)[6U] = ((*v_3)[6U]) | (((_S198 >> (bitpos_16 & 7U)) & (low_mask_0(eb_3))) << _S155);
        }
        uint index_9 = rank_3 + ((uint)popcount((flags_1 & (low_mask_0(7U)))));
        if(((flags_1 >> 7U) & 1U) != 0U)
        {
            _S191 = index_9 < _S156;
        }
        else
        {
            _S191 = false;
        }
        if(_S191)
        {
            uint bitpos_17 = index_9 * eb_3;
            uint _S199 = load32_0(excStart_1 + (bitpos_17 >> 3U), kernelContext_15);
            (*v_3)[7U] = ((*v_3)[7U]) | (((_S199 >> (bitpos_17 & 7U)) & (low_mask_0(eb_3))) << _S155);
        }
    }
    return;
}

uint unzigzag_0(uint v_4)
{
    return ((v_4 >> 1U) ^ (0U - (v_4 & 1U))) & (low_mask_0(8U));
}

void block_planar_0(uint t_6, uint h_0, array<uint, int(8)> thread* v_5)
{
    uint _S200 = (*v_5)[1U] + (*v_5)[0U];
    (*v_5)[1U] = _S200;
    uint _S201 = (*v_5)[2U] + _S200;
    (*v_5)[2U] = _S201;
    uint _S202 = (*v_5)[3U] + _S201;
    (*v_5)[3U] = _S202;
    uint _S203 = (*v_5)[4U] + _S202;
    (*v_5)[4U] = _S203;
    uint _S204 = (*v_5)[5U] + _S203;
    (*v_5)[5U] = _S204;
    uint _S205 = (*v_5)[6U] + _S204;
    (*v_5)[6U] = _S205;
    (*v_5)[7U] = (*v_5)[7U] + _S205;
    uint d_0;
    if(h_0 == 16U)
    {
        uint _S206 = (simd_shuffle_xor(((*v_5)[7U]), ushort((16U))));
        bool _S207 = t_6 >= 16U;
        if(_S207)
        {
            (*v_5)[0U] = (*v_5)[0U] + _S206;
            (*v_5)[1U] = (*v_5)[1U] + _S206;
            (*v_5)[2U] = (*v_5)[2U] + _S206;
            (*v_5)[3U] = (*v_5)[3U] + _S206;
            (*v_5)[4U] = (*v_5)[4U] + _S206;
            (*v_5)[5U] = (*v_5)[5U] + _S206;
            (*v_5)[6U] = (*v_5)[6U] + _S206;
            (*v_5)[7U] = (*v_5)[7U] + _S206;
        }
        uint p_1 = (simd_prefix_inclusive_sum(((*v_5)[0U])));
        uint top_0 = (simd_broadcast((p_1), ushort((15U))));
        if(_S207)
        {
            d_0 = p_1 - top_0;
        }
        else
        {
            d_0 = p_1;
        }
        (*v_5)[0U] = d_0;
        uint p_2 = (simd_prefix_inclusive_sum(((*v_5)[1U])));
        uint top_1 = (simd_broadcast((p_2), ushort((15U))));
        if(_S207)
        {
            d_0 = p_2 - top_1;
        }
        else
        {
            d_0 = p_2;
        }
        (*v_5)[1U] = d_0;
        uint p_3 = (simd_prefix_inclusive_sum(((*v_5)[2U])));
        uint top_2 = (simd_broadcast((p_3), ushort((15U))));
        if(_S207)
        {
            d_0 = p_3 - top_2;
        }
        else
        {
            d_0 = p_3;
        }
        (*v_5)[2U] = d_0;
        uint p_4 = (simd_prefix_inclusive_sum(((*v_5)[3U])));
        uint top_3 = (simd_broadcast((p_4), ushort((15U))));
        if(_S207)
        {
            d_0 = p_4 - top_3;
        }
        else
        {
            d_0 = p_4;
        }
        (*v_5)[3U] = d_0;
        uint p_5 = (simd_prefix_inclusive_sum(((*v_5)[4U])));
        uint top_4 = (simd_broadcast((p_5), ushort((15U))));
        if(_S207)
        {
            d_0 = p_5 - top_4;
        }
        else
        {
            d_0 = p_5;
        }
        (*v_5)[4U] = d_0;
        uint p_6 = (simd_prefix_inclusive_sum(((*v_5)[5U])));
        uint top_5 = (simd_broadcast((p_6), ushort((15U))));
        if(_S207)
        {
            d_0 = p_6 - top_5;
        }
        else
        {
            d_0 = p_6;
        }
        (*v_5)[5U] = d_0;
        uint p_7 = (simd_prefix_inclusive_sum(((*v_5)[6U])));
        uint top_6 = (simd_broadcast((p_7), ushort((15U))));
        if(_S207)
        {
            d_0 = p_7 - top_6;
        }
        else
        {
            d_0 = p_7;
        }
        (*v_5)[6U] = d_0;
        uint p_8 = (simd_prefix_inclusive_sum(((*v_5)[7U])));
        uint top_7 = (simd_broadcast((p_8), ushort((15U))));
        if(_S207)
        {
            d_0 = p_8 - top_7;
        }
        else
        {
            d_0 = p_8;
        }
        (*v_5)[7U] = d_0;
        return;
    }
    uint total_17 = (*v_5)[7U];
    d_0 = h_0;
    uint inclusive_0 = (*v_5)[7U];
    for(;;)
    {
        if(d_0 < 32U)
        {
        }
        else
        {
            break;
        }
        uint u_0 = (simd_shuffle_up((inclusive_0), ushort((d_0))));
        if(t_6 >= d_0)
        {
            inclusive_0 = inclusive_0 + u_0;
        }
        d_0 = d_0 * 2U;
    }
    uint _S208 = inclusive_0 - total_17;
    (*v_5)[0U] = (*v_5)[0U] + _S208;
    (*v_5)[1U] = (*v_5)[1U] + _S208;
    (*v_5)[2U] = (*v_5)[2U] + _S208;
    (*v_5)[3U] = (*v_5)[3U] + _S208;
    (*v_5)[4U] = (*v_5)[4U] + _S208;
    (*v_5)[5U] = (*v_5)[5U] + _S208;
    (*v_5)[6U] = (*v_5)[6U] + _S208;
    (*v_5)[7U] = (*v_5)[7U] + _S208;
    uint _S209 = t_6 % h_0;
    d_0 = 1U;
    for(;;)
    {
        if(d_0 < h_0)
        {
        }
        else
        {
            break;
        }
        uint u_1 = (simd_shuffle_up(((*v_5)[0U]), ushort((d_0))));
        bool _S210 = _S209 >= d_0;
        if(_S210)
        {
            (*v_5)[0U] = (*v_5)[0U] + u_1;
        }
        uint u_2 = (simd_shuffle_up(((*v_5)[1U]), ushort((d_0))));
        if(_S210)
        {
            (*v_5)[1U] = (*v_5)[1U] + u_2;
        }
        uint u_3 = (simd_shuffle_up(((*v_5)[2U]), ushort((d_0))));
        if(_S210)
        {
            (*v_5)[2U] = (*v_5)[2U] + u_3;
        }
        uint u_4 = (simd_shuffle_up(((*v_5)[3U]), ushort((d_0))));
        if(_S210)
        {
            (*v_5)[3U] = (*v_5)[3U] + u_4;
        }
        uint u_5 = (simd_shuffle_up(((*v_5)[4U]), ushort((d_0))));
        if(_S210)
        {
            (*v_5)[4U] = (*v_5)[4U] + u_5;
        }
        uint u_6 = (simd_shuffle_up(((*v_5)[5U]), ushort((d_0))));
        if(_S210)
        {
            (*v_5)[5U] = (*v_5)[5U] + u_6;
        }
        uint u_7 = (simd_shuffle_up(((*v_5)[6U]), ushort((d_0))));
        if(_S210)
        {
            (*v_5)[6U] = (*v_5)[6U] + u_7;
        }
        uint u_8 = (simd_shuffle_up(((*v_5)[7U]), ushort((d_0))));
        if(_S210)
        {
            (*v_5)[7U] = (*v_5)[7U] + u_8;
        }
        d_0 = d_0 * 2U;
    }
    return;
}

void store_block_row_0(uint elt_3, uint n_16, const array<uint, int(8)> thread* v_6, KernelContext_0 thread* kernelContext_16)
{
    bool _S211;
    if(n_16 == 8U)
    {
        _S211 = aligned_run_0(elt_3);
    }
    else
    {
        _S211 = false;
    }
    if(_S211)
    {
        store_run_0(elt_3, v_6, kernelContext_16);
        return;
    }
    if(0U < n_16)
    {
        *(kernelContext_16->output_0+elt_3) = uchar((*v_6)[0U]);
    }
    if(1U < n_16)
    {
        *(kernelContext_16->output_0+(elt_3 + 1U)) = uchar((*v_6)[1U]);
    }
    if(2U < n_16)
    {
        *(kernelContext_16->output_0+(elt_3 + 2U)) = uchar((*v_6)[2U]);
    }
    if(3U < n_16)
    {
        *(kernelContext_16->output_0+(elt_3 + 3U)) = uchar((*v_6)[3U]);
    }
    if(4U < n_16)
    {
        *(kernelContext_16->output_0+(elt_3 + 4U)) = uchar((*v_6)[4U]);
    }
    if(5U < n_16)
    {
        *(kernelContext_16->output_0+(elt_3 + 5U)) = uchar((*v_6)[5U]);
    }
    if(6U < n_16)
    {
        *(kernelContext_16->output_0+(elt_3 + 6U)) = uchar((*v_6)[6U]);
    }
    if(7U < n_16)
    {
        *(kernelContext_16->output_0+(elt_3 + 7U)) = uchar((*v_6)[7U]);
    }
    return;
}

void decode_strip_0(uint index_10, uint t_7, KernelContext_0 thread* kernelContext_17)
{
    strip_task_0 strip_0 = kernelContext_17->strips_0[index_10];
    if((strip_0.blockCount_1) == 0U)
    {
        return;
    }
    uint _S212 = 256U / strip_0.height_0;
    uint _S213 = t_7 % strip_0.height_0;
    uint _S214 = t_7 / strip_0.height_0;
    uint _S215 = _S214 * 8U;
    thread bool bad_6 = false;
    uint k_3 = 0U;
    uint carry_1 = 0U;
    for(;;)
    {
        if(k_3 < (strip_0.blockCount_1))
        {
        }
        else
        {
            break;
        }
        block_ref_0 device* _S216 = kernelContext_17->bandBlocks_0+(strip_0.firstBlock_1 + k_3);
        block_ref_0 ref_1 = *_S216;
        thread array<uint, int(8)> v_7;
        if(((*_S216).rest_0) == 4294967295U)
        {
            v_7[0U] = 0U;
            v_7[1U] = 0U;
            v_7[2U] = 0U;
            v_7[3U] = 0U;
            v_7[4U] = 0U;
            v_7[5U] = 0U;
            v_7[6U] = 0U;
            v_7[7U] = 0U;
        }
        else
        {
            block_lane_0(ref_1.hdr_0, ref_1.rest_0, t_7, &v_7, &bad_6, kernelContext_17);
            v_7[0U] = unzigzag_0(v_7[0U]);
            v_7[1U] = unzigzag_0(v_7[1U]);
            v_7[2U] = unzigzag_0(v_7[2U]);
            v_7[3U] = unzigzag_0(v_7[3U]);
            v_7[4U] = unzigzag_0(v_7[4U]);
            v_7[5U] = unzigzag_0(v_7[5U]);
            v_7[6U] = unzigzag_0(v_7[6U]);
            v_7[7U] = unzigzag_0(v_7[7U]);
        }
        block_planar_0(t_7, strip_0.height_0, &v_7);
        v_7[0U] = v_7[0U] + carry_1;
        v_7[1U] = v_7[1U] + carry_1;
        v_7[2U] = v_7[2U] + carry_1;
        v_7[3U] = v_7[3U] + carry_1;
        v_7[4U] = v_7[4U] + carry_1;
        v_7[5U] = v_7[5U] + carry_1;
        v_7[6U] = v_7[6U] + carry_1;
        uint _S217 = v_7[7U] + carry_1;
        v_7[7U] = _S217;
        uint _S218 = (simd_shuffle((_S217), ushort((_S213 + 32U - strip_0.height_0))));
        uint x0_0 = k_3 * _S212;
        uint _S219 = min(_S212, strip_0.columns_0 - x0_0);
        if(_S215 < _S219)
        {
            uint _S220 = strip_0.outBase_1 + _S213 * strip_0.width_0 + x0_0 + _S215;
            uint _S221 = min(8U, _S219 - _S215);
            thread array<uint, int(8)> _S222 = v_7;
            store_block_row_0(_S220, _S221, &_S222, kernelContext_17);
        }
        k_3 = k_3 + 1U;
        carry_1 = _S218;
    }
    bool _S223 = simd_any(bad_6);
    bool _S224;
    if(_S223)
    {
        _S224 = t_7 == 0U;
    }
    else
    {
        _S224 = false;
    }
    if(_S224)
    {
        report_0(1U, strip_0.frame_2, kernelContext_17);
    }
    return;
}

[[kernel]] void decode_strips(uint3 group_3 [[threadgroup_position_in_grid]], uint3 thread_3 [[thread_position_in_threadgroup]], block_task_0 device* blocks_4 [[buffer(1)]], uint32_t device* payload_4 [[buffer(0)]], uint32_t device* outputBytes_4 [[buffer(3)]], uchar device* output_4 [[buffer(2)]], block_chunk_0 device* blockChunks_4 [[buffer(8)]], band_frame_0 device* bandFrames_4 [[buffer(4)]], uint device* chunkSizes_4 [[buffer(9)]], block_ref_0 device* bandBlocks_4 [[buffer(6)]], uint device* status_4 [[buffer(7)]], strip_task_0 device* strips_4 [[buffer(5)]], raw_copy_0 device* rawCopies_4 [[buffer(11)]], uchar device* rawValues_4 [[buffer(10)]])
{
    thread KernelContext_0 kernelContext_18;
    (&kernelContext_18)->blocks_0 = blocks_4;
    (&kernelContext_18)->payload_0 = payload_4;
    (&kernelContext_18)->outputBytes_0 = outputBytes_4;
    (&kernelContext_18)->output_0 = output_4;
    (&kernelContext_18)->blockChunks_0 = blockChunks_4;
    (&kernelContext_18)->bandFrames_0 = bandFrames_4;
    (&kernelContext_18)->chunkSizes_0 = chunkSizes_4;
    (&kernelContext_18)->bandBlocks_0 = bandBlocks_4;
    (&kernelContext_18)->status_0 = status_4;
    (&kernelContext_18)->strips_0 = strips_4;
    (&kernelContext_18)->rawCopies_0 = rawCopies_4;
    (&kernelContext_18)->rawValues_0 = rawValues_4;
    uint _S225 = thread_3.x;
    decode_strip_0(group_3.x * 4U + _S225 / 32U, _S225 % 32U, &kernelContext_18);
    return;
}

[[kernel]] void copy_raw(uint3 group_4 [[threadgroup_position_in_grid]], uint3 thread_4 [[thread_position_in_threadgroup]], block_task_0 device* blocks_5 [[buffer(1)]], uint32_t device* payload_5 [[buffer(0)]], uint32_t device* outputBytes_5 [[buffer(3)]], uchar device* output_5 [[buffer(2)]], block_chunk_0 device* blockChunks_5 [[buffer(8)]], band_frame_0 device* bandFrames_5 [[buffer(4)]], uint device* chunkSizes_5 [[buffer(9)]], block_ref_0 device* bandBlocks_5 [[buffer(6)]], uint device* status_5 [[buffer(7)]], strip_task_0 device* strips_5 [[buffer(5)]], raw_copy_0 device* rawCopies_5 [[buffer(11)]], uchar device* rawValues_5 [[buffer(10)]])
{
    thread KernelContext_0 kernelContext_19;
    (&kernelContext_19)->blocks_0 = blocks_5;
    (&kernelContext_19)->payload_0 = payload_5;
    (&kernelContext_19)->outputBytes_0 = outputBytes_5;
    (&kernelContext_19)->output_0 = output_5;
    (&kernelContext_19)->blockChunks_0 = blockChunks_5;
    (&kernelContext_19)->bandFrames_0 = bandFrames_5;
    (&kernelContext_19)->chunkSizes_0 = chunkSizes_5;
    (&kernelContext_19)->bandBlocks_0 = bandBlocks_5;
    (&kernelContext_19)->status_0 = status_5;
    (&kernelContext_19)->strips_0 = strips_5;
    (&kernelContext_19)->rawCopies_0 = rawCopies_5;
    (&kernelContext_19)->rawValues_0 = rawValues_5;
    raw_copy_0 _S226 = rawCopies_5[group_4.x];
    uint i_2 = thread_4.x;
    for(;;)
    {
        if(i_2 < (_S226.count_2))
        {
        }
        else
        {
            break;
        }
        *((&kernelContext_19)->output_0+(_S226.destination_0 + i_2)) = (&kernelContext_19)->rawValues_0[_S226.source_0 + i_2];
        i_2 = i_2 + 128U;
    }
    return;
}

