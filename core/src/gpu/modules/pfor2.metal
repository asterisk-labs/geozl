#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct block_task_0
{
    uint payOff_0;
    uint outBase_0;
    uint count_0;
    uint transform_0;
};

struct plane_task_0
{
    uint outBase_1;
    uint width_0;
    uint rows_0;
};

struct raw_copy_0
{
    uint source_0;
    uint destination_0;
    uint count_1;
};

struct KernelContext_0
{
    block_task_0 device* blocks_0;
    uint32_t device* payload_0;
    uint32_t device* outputBytes_0;
    ushort device* output_0;
    plane_task_0 device* planes_0;
    raw_copy_0 device* rawCopies_0;
    ushort device* rawValues_0;
};

uint load8_0(uint a_0, KernelContext_0 thread* kernelContext_0)
{
    uint _S1 = as_type<uint>(kernelContext_0->payload_0[(uint(int(a_0 & 4294967292U)))>>2]);
    return (_S1 >> ((a_0 & 3U) * 8U)) & 255U;
}

uint load32_0(uint a_1, KernelContext_0 thread* kernelContext_1)
{
    uint shift_0 = (a_1 & 3U) * 8U;
    uint _S2 = a_1 & 4294967292U;
    uint _S3 = as_type<uint>(kernelContext_1->payload_0[(uint(int(_S2)))>>2]);
    if(shift_0 == 0U)
    {
        return _S3;
    }
    uint _S4 = as_type<uint>(kernelContext_1->payload_0[(uint(int(_S2 + 4U)))>>2]);
    return (_S3 >> shift_0) | (_S4 << (32U - shift_0));
}

uint low_mask_0(uint b_0)
{
    uint _S5;
    if(b_0 >= 32U)
    {
        _S5 = 4294967295U;
    }
    else
    {
        _S5 = (1U << b_0) - 1U;
    }
    return _S5;
}

void load128_0(uint a_2, array<uint, int(4)> thread* w_0, KernelContext_0 thread* kernelContext_2)
{
    uint base_0 = a_2 & 4294967292U;
    uint shift_1 = (a_2 & 3U) * 8U;
    uint _S6 = as_type<uint>(kernelContext_2->payload_0[(base_0)>>2]);
    uint _S7 = as_type<uint>(kernelContext_2->payload_0[(base_0 + 4U)>>2]);
    uint _S8 = as_type<uint>(kernelContext_2->payload_0[(base_0 + 8U)>>2]);
    uint _S9 = as_type<uint>(kernelContext_2->payload_0[(base_0 + 12U)>>2]);
    uint _S10 = as_type<uint>(kernelContext_2->payload_0[(uint(int(base_0 + 16U)))>>2]);
    if(shift_1 == 0U)
    {
        (*w_0)[int(0)] = _S6;
        (*w_0)[int(1)] = _S7;
        (*w_0)[int(2)] = _S8;
        (*w_0)[int(3)] = _S9;
        return;
    }
    uint _S11 = 32U - shift_1;
    (*w_0)[int(0)] = (_S6 >> shift_1) | (_S7 << _S11);
    (*w_0)[int(1)] = (_S7 >> shift_1) | (_S8 << _S11);
    (*w_0)[int(2)] = (_S8 >> shift_1) | (_S9 << _S11);
    (*w_0)[int(3)] = (_S9 >> shift_1) | (_S10 << _S11);
    return;
}

uint body_word_0(const array<uint, int(8)> thread* row_0, uint c_0)
{
    return (((*row_0)[c_0 >> 1U]) >> ((c_0 & 1U) * 16U)) & 65535U;
}

void pfor_lane_0(uint base_1, uint t_0, array<uint, int(8)> thread* v_0, KernelContext_0 thread* kernelContext_3)
{
    uint _S12 = load8_0(base_1, kernelContext_3);
    uint _S13 = load8_0(base_1 + 1U, kernelContext_3);
    thread array<uint, int(8)> bitmap_0;
    bitmap_0[0U] = 0U;
    bitmap_0[1U] = 0U;
    bitmap_0[2U] = 0U;
    bitmap_0[3U] = 0U;
    bitmap_0[4U] = 0U;
    bitmap_0[5U] = 0U;
    bitmap_0[6U] = 0U;
    bitmap_0[7U] = 0U;
    uint _S14 = base_1 + 2U;
    uint body_0;
    uint eb_0;
    uint excStart_0;
    uint rank_0;
    uint off_0;
    if(_S13 != 0U)
    {
        uint _S15 = load8_0(_S14, kernelContext_3);
        uint eb_1 = _S15 & 127U;
        uint posStart_0 = base_1 + 3U;
        if((_S15 >> 7U) == 0U)
        {
            if(t_0 < 8U)
            {
                uint _S16 = load32_0(posStart_0 + t_0 * 4U, kernelContext_3);
                body_0 = _S16;
            }
            else
            {
                body_0 = 0U;
            }
            uint _S17 = (simd_shuffle((body_0), ushort((int(0)))));
            bitmap_0[0U] = _S17;
            uint _S18 = (simd_shuffle((body_0), ushort((int(1)))));
            bitmap_0[1U] = _S18;
            uint _S19 = (simd_shuffle((body_0), ushort((int(2)))));
            bitmap_0[2U] = _S19;
            uint _S20 = (simd_shuffle((body_0), ushort((int(3)))));
            bitmap_0[3U] = _S20;
            uint _S21 = (simd_shuffle((body_0), ushort((int(4)))));
            bitmap_0[4U] = _S21;
            uint _S22 = (simd_shuffle((body_0), ushort((int(5)))));
            bitmap_0[5U] = _S22;
            uint _S23 = (simd_shuffle((body_0), ushort((int(6)))));
            bitmap_0[6U] = _S23;
            uint _S24 = (simd_shuffle((body_0), ushort((int(7)))));
            bitmap_0[7U] = _S24;
            body_0 = posStart_0 + 32U;
        }
        else
        {
            uint _S25 = (_S13 + 32U - 1U) / 32U;
            body_0 = 0U;
            for(;;)
            {
                if(body_0 < _S25)
                {
                }
                else
                {
                    break;
                }
                uint i_0 = body_0 * 32U + t_0;
                if(i_0 < _S13)
                {
                    uint _S26 = load8_0(posStart_0 + i_0, kernelContext_3);
                    eb_0 = _S26;
                }
                else
                {
                    eb_0 = 65535U;
                }
                uint _S27 = eb_0 >> 5U;
                if(_S27 == 0U)
                {
                    excStart_0 = 1U << (eb_0 & 31U);
                }
                else
                {
                    excStart_0 = 0U;
                }
                uint _S28 = simd_or(excStart_0);
                bitmap_0[0U] = (bitmap_0[0U]) | _S28;
                if(_S27 == 1U)
                {
                    rank_0 = 1U << (eb_0 & 31U);
                }
                else
                {
                    rank_0 = 0U;
                }
                uint _S29 = simd_or(rank_0);
                bitmap_0[1U] = (bitmap_0[1U]) | _S29;
                if(_S27 == 2U)
                {
                    off_0 = 1U << (eb_0 & 31U);
                }
                else
                {
                    off_0 = 0U;
                }
                uint _S30 = simd_or(off_0);
                bitmap_0[2U] = (bitmap_0[2U]) | _S30;
                uint _S31;
                if(_S27 == 3U)
                {
                    _S31 = 1U << (eb_0 & 31U);
                }
                else
                {
                    _S31 = 0U;
                }
                uint _S32 = simd_or(_S31);
                bitmap_0[3U] = (bitmap_0[3U]) | _S32;
                uint _S33;
                if(_S27 == 4U)
                {
                    _S33 = 1U << (eb_0 & 31U);
                }
                else
                {
                    _S33 = 0U;
                }
                uint _S34 = simd_or(_S33);
                bitmap_0[4U] = (bitmap_0[4U]) | _S34;
                uint _S35;
                if(_S27 == 5U)
                {
                    _S35 = 1U << (eb_0 & 31U);
                }
                else
                {
                    _S35 = 0U;
                }
                uint _S36 = simd_or(_S35);
                bitmap_0[5U] = (bitmap_0[5U]) | _S36;
                uint _S37;
                if(_S27 == 6U)
                {
                    _S37 = 1U << (eb_0 & 31U);
                }
                else
                {
                    _S37 = 0U;
                }
                uint _S38 = simd_or(_S37);
                bitmap_0[6U] = (bitmap_0[6U]) | _S38;
                uint _S39;
                if(_S27 == 7U)
                {
                    _S39 = 1U << (eb_0 & 31U);
                }
                else
                {
                    _S39 = 0U;
                }
                uint _S40 = simd_or(_S39);
                bitmap_0[7U] = (bitmap_0[7U]) | _S40;
                body_0 = body_0 + 1U;
            }
            body_0 = posStart_0 + _S13;
        }
        uint _S41 = body_0;
        body_0 = body_0 + ((_S13 * eb_1 + 7U) >> 3U);
        eb_0 = eb_1;
        excStart_0 = _S41;
    }
    else
    {
        body_0 = _S14;
        eb_0 = 0U;
        excStart_0 = _S14;
    }
    uint i0_0 = t_0 * 8U;
    uint _S42 = t_0 >> 2U;
    uint _S43 = (t_0 & 3U) * 8U;
    uint _S44 = ((bitmap_0[_S42]) >> _S43) & 255U;
    if(0U < _S42)
    {
        rank_0 = ((uint)popcount((bitmap_0[0U])));
    }
    else
    {
        rank_0 = 0U;
    }
    if(1U < _S42)
    {
        rank_0 = rank_0 + ((uint)popcount((bitmap_0[1U])));
    }
    if(2U < _S42)
    {
        rank_0 = rank_0 + ((uint)popcount((bitmap_0[2U])));
    }
    if(3U < _S42)
    {
        rank_0 = rank_0 + ((uint)popcount((bitmap_0[3U])));
    }
    if(4U < _S42)
    {
        rank_0 = rank_0 + ((uint)popcount((bitmap_0[4U])));
    }
    if(5U < _S42)
    {
        rank_0 = rank_0 + ((uint)popcount((bitmap_0[5U])));
    }
    if(6U < _S42)
    {
        rank_0 = rank_0 + ((uint)popcount((bitmap_0[6U])));
    }
    if(7U < _S42)
    {
        rank_0 = rank_0 + ((uint)popcount((bitmap_0[7U])));
    }
    uint rank_1 = rank_0 + ((uint)popcount(((bitmap_0[_S42]) & (low_mask_0(_S43)))));
    thread array<uint, int(8)> row_1;
    row_1[0U] = 0U;
    row_1[1U] = 0U;
    row_1[2U] = 0U;
    row_1[3U] = 0U;
    row_1[4U] = 0U;
    row_1[5U] = 0U;
    row_1[6U] = 0U;
    row_1[7U] = 0U;
    if(_S12 != 0U)
    {
        uint bitpos_0 = i0_0 / 8U * _S12;
        uint off_1 = bitpos_0 % 16U;
        uint a_3 = body_0 + bitpos_0 / 16U * 16U + i0_0 % 8U * 2U;
        thread array<uint, int(4)> w_1;
        load128_0(a_3, &w_1, kernelContext_3);
        row_1[int(0)] = w_1[int(0)];
        row_1[int(1)] = w_1[int(1)];
        row_1[int(2)] = w_1[int(2)];
        row_1[int(3)] = w_1[int(3)];
        if((off_1 + _S12) > 16U)
        {
            load128_0(a_3 + 16U, &w_1, kernelContext_3);
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
    uint _S45 = low_mask_0(_S12);
    thread array<uint, int(8)> _S46 = row_1;
    uint _S47 = body_word_0(&_S46, 0U);
    thread array<uint, int(8)> _S48 = row_1;
    uint _S49 = body_word_0(&_S48, 8U);
    (*v_0)[0U] = ((_S47 | (_S49 << 16U)) >> off_0) & _S45;
    thread array<uint, int(8)> _S50 = row_1;
    uint _S51 = body_word_0(&_S50, 1U);
    thread array<uint, int(8)> _S52 = row_1;
    uint _S53 = body_word_0(&_S52, 9U);
    (*v_0)[1U] = ((_S51 | (_S53 << 16U)) >> off_0) & _S45;
    thread array<uint, int(8)> _S54 = row_1;
    uint _S55 = body_word_0(&_S54, 2U);
    thread array<uint, int(8)> _S56 = row_1;
    uint _S57 = body_word_0(&_S56, 10U);
    (*v_0)[2U] = ((_S55 | (_S57 << 16U)) >> off_0) & _S45;
    thread array<uint, int(8)> _S58 = row_1;
    uint _S59 = body_word_0(&_S58, 3U);
    thread array<uint, int(8)> _S60 = row_1;
    uint _S61 = body_word_0(&_S60, 11U);
    (*v_0)[3U] = ((_S59 | (_S61 << 16U)) >> off_0) & _S45;
    thread array<uint, int(8)> _S62 = row_1;
    uint _S63 = body_word_0(&_S62, 4U);
    thread array<uint, int(8)> _S64 = row_1;
    uint _S65 = body_word_0(&_S64, 12U);
    (*v_0)[4U] = ((_S63 | (_S65 << 16U)) >> off_0) & _S45;
    thread array<uint, int(8)> _S66 = row_1;
    uint _S67 = body_word_0(&_S66, 5U);
    thread array<uint, int(8)> _S68 = row_1;
    uint _S69 = body_word_0(&_S68, 13U);
    (*v_0)[5U] = ((_S67 | (_S69 << 16U)) >> off_0) & _S45;
    thread array<uint, int(8)> _S70 = row_1;
    uint _S71 = body_word_0(&_S70, 6U);
    thread array<uint, int(8)> _S72 = row_1;
    uint _S73 = body_word_0(&_S72, 14U);
    (*v_0)[6U] = ((_S71 | (_S73 << 16U)) >> off_0) & _S45;
    thread array<uint, int(8)> _S74 = row_1;
    uint _S75 = body_word_0(&_S74, 7U);
    thread array<uint, int(8)> _S76 = row_1;
    uint _S77 = body_word_0(&_S76, 15U);
    (*v_0)[7U] = ((_S75 | (_S77 << 16U)) >> off_0) & _S45;
    if(((_S44 >> 0U) & 1U) != 0U)
    {
        uint bitpos_1 = (rank_1 + ((uint)popcount((_S44 & (low_mask_0(0U)))))) * eb_0;
        uint _S78 = load32_0(excStart_0 + (bitpos_1 >> 3U), kernelContext_3);
        (*v_0)[0U] = ((*v_0)[0U]) | (((_S78 >> (bitpos_1 & 7U)) & (low_mask_0(eb_0))) << _S12);
    }
    if(((_S44 >> 1U) & 1U) != 0U)
    {
        uint bitpos_2 = (rank_1 + ((uint)popcount((_S44 & (low_mask_0(1U)))))) * eb_0;
        uint _S79 = load32_0(excStart_0 + (bitpos_2 >> 3U), kernelContext_3);
        (*v_0)[1U] = ((*v_0)[1U]) | (((_S79 >> (bitpos_2 & 7U)) & (low_mask_0(eb_0))) << _S12);
    }
    if(((_S44 >> 2U) & 1U) != 0U)
    {
        uint bitpos_3 = (rank_1 + ((uint)popcount((_S44 & (low_mask_0(2U)))))) * eb_0;
        uint _S80 = load32_0(excStart_0 + (bitpos_3 >> 3U), kernelContext_3);
        (*v_0)[2U] = ((*v_0)[2U]) | (((_S80 >> (bitpos_3 & 7U)) & (low_mask_0(eb_0))) << _S12);
    }
    if(((_S44 >> 3U) & 1U) != 0U)
    {
        uint bitpos_4 = (rank_1 + ((uint)popcount((_S44 & (low_mask_0(3U)))))) * eb_0;
        uint _S81 = load32_0(excStart_0 + (bitpos_4 >> 3U), kernelContext_3);
        (*v_0)[3U] = ((*v_0)[3U]) | (((_S81 >> (bitpos_4 & 7U)) & (low_mask_0(eb_0))) << _S12);
    }
    if(((_S44 >> 4U) & 1U) != 0U)
    {
        uint bitpos_5 = (rank_1 + ((uint)popcount((_S44 & (low_mask_0(4U)))))) * eb_0;
        uint _S82 = load32_0(excStart_0 + (bitpos_5 >> 3U), kernelContext_3);
        (*v_0)[4U] = ((*v_0)[4U]) | (((_S82 >> (bitpos_5 & 7U)) & (low_mask_0(eb_0))) << _S12);
    }
    if(((_S44 >> 5U) & 1U) != 0U)
    {
        uint bitpos_6 = (rank_1 + ((uint)popcount((_S44 & (low_mask_0(5U)))))) * eb_0;
        uint _S83 = load32_0(excStart_0 + (bitpos_6 >> 3U), kernelContext_3);
        (*v_0)[5U] = ((*v_0)[5U]) | (((_S83 >> (bitpos_6 & 7U)) & (low_mask_0(eb_0))) << _S12);
    }
    if(((_S44 >> 6U) & 1U) != 0U)
    {
        uint bitpos_7 = (rank_1 + ((uint)popcount((_S44 & (low_mask_0(6U)))))) * eb_0;
        uint _S84 = load32_0(excStart_0 + (bitpos_7 >> 3U), kernelContext_3);
        (*v_0)[6U] = ((*v_0)[6U]) | (((_S84 >> (bitpos_7 & 7U)) & (low_mask_0(eb_0))) << _S12);
    }
    if(((_S44 >> 7U) & 1U) != 0U)
    {
        uint bitpos_8 = (rank_1 + ((uint)popcount((_S44 & (low_mask_0(7U)))))) * eb_0;
        uint _S85 = load32_0(excStart_0 + (bitpos_8 >> 3U), kernelContext_3);
        (*v_0)[7U] = ((*v_0)[7U]) | (((_S85 >> (bitpos_8 & 7U)) & (low_mask_0(eb_0))) << _S12);
    }
    return;
}

uint unzigzag_0(uint v_1)
{
    return ((v_1 >> 1U) ^ (0U - (v_1 & 1U))) & (low_mask_0(16U));
}

bool aligned_run_0(uint elt_0)
{
    return ((elt_0 * 2U) & 15U) == 0U;
}

void store_run_0(uint elt_1, const array<uint, int(8)> thread* v_2, KernelContext_0 thread* kernelContext_4)
{
    uint _S86 = elt_1 * 2U;
    uint _S87 = (((*v_2)[int(2)]) & 65535U) | (((*v_2)[int(3)]) << 16U);
    uint _S88 = (((*v_2)[int(4)]) & 65535U) | (((*v_2)[int(5)]) << 16U);
    uint _S89 = (((*v_2)[int(6)]) & 65535U) | (((*v_2)[int(7)]) << 16U);
    kernelContext_4->outputBytes_0[(_S86)>>2] = as_type<uint32_t>((((*v_2)[int(0)]) & 65535U) | (((*v_2)[int(1)]) << 16U));
    kernelContext_4->outputBytes_0[(_S86 + 4U)>>2] = as_type<uint32_t>(_S87);
    kernelContext_4->outputBytes_0[(_S86 + 8U)>>2] = as_type<uint32_t>(_S88);
    kernelContext_4->outputBytes_0[(_S86 + 12U)>>2] = as_type<uint32_t>(_S89);
    return;
}

[[kernel]] void unpack_blocks(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_0 [[thread_position_in_threadgroup]], block_task_0 device* blocks_1 [[buffer(1)]], uint32_t device* payload_1 [[buffer(0)]], uint32_t device* outputBytes_1 [[buffer(4)]], ushort device* output_1 [[buffer(3)]], plane_task_0 device* planes_1 [[buffer(2)]], raw_copy_0 device* rawCopies_1 [[buffer(6)]], ushort device* rawValues_1 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_5;
    (&kernelContext_5)->blocks_0 = blocks_1;
    (&kernelContext_5)->payload_0 = payload_1;
    (&kernelContext_5)->outputBytes_0 = outputBytes_1;
    (&kernelContext_5)->output_0 = output_1;
    (&kernelContext_5)->planes_0 = planes_1;
    (&kernelContext_5)->rawCopies_0 = rawCopies_1;
    (&kernelContext_5)->rawValues_0 = rawValues_1;
    uint _S90 = thread_0.x;
    uint t_1 = _S90 % 32U;
    block_task_0 task_0 = blocks_1[group_0.x * 4U + _S90 / 32U];
    thread array<uint, int(8)> v_3;
    pfor_lane_0(task_0.payOff_0, t_1, &v_3, &kernelContext_5);
    if((task_0.transform_0) != 0U)
    {
        v_3[0U] = unzigzag_0(v_3[0U]);
        v_3[1U] = unzigzag_0(v_3[1U]);
        v_3[2U] = unzigzag_0(v_3[2U]);
        v_3[3U] = unzigzag_0(v_3[3U]);
        v_3[4U] = unzigzag_0(v_3[4U]);
        v_3[5U] = unzigzag_0(v_3[5U]);
        v_3[6U] = unzigzag_0(v_3[6U]);
        v_3[7U] = unzigzag_0(v_3[7U]);
    }
    uint i0_1 = t_1 * 8U;
    uint elt_2 = task_0.outBase_0 + i0_1;
    bool _S91;
    if((i0_1 + 8U) <= (task_0.count_0))
    {
        _S91 = aligned_run_0(elt_2);
    }
    else
    {
        _S91 = false;
    }
    if(_S91)
    {
        thread array<uint, int(8)> _S92 = v_3;
        store_run_0(elt_2, &_S92, &kernelContext_5);
    }
    else
    {
        if(i0_1 < (task_0.count_0))
        {
            *((&kernelContext_5)->output_0+elt_2) = ushort(v_3[0U]);
        }
        if((i0_1 + 1U) < (task_0.count_0))
        {
            *((&kernelContext_5)->output_0+(elt_2 + 1U)) = ushort(v_3[1U]);
        }
        if((i0_1 + 2U) < (task_0.count_0))
        {
            *((&kernelContext_5)->output_0+(elt_2 + 2U)) = ushort(v_3[2U]);
        }
        if((i0_1 + 3U) < (task_0.count_0))
        {
            *((&kernelContext_5)->output_0+(elt_2 + 3U)) = ushort(v_3[3U]);
        }
        if((i0_1 + 4U) < (task_0.count_0))
        {
            *((&kernelContext_5)->output_0+(elt_2 + 4U)) = ushort(v_3[4U]);
        }
        if((i0_1 + 5U) < (task_0.count_0))
        {
            *((&kernelContext_5)->output_0+(elt_2 + 5U)) = ushort(v_3[5U]);
        }
        if((i0_1 + 6U) < (task_0.count_0))
        {
            *((&kernelContext_5)->output_0+(elt_2 + 6U)) = ushort(v_3[6U]);
        }
        if((i0_1 + 7U) < (task_0.count_0))
        {
            *((&kernelContext_5)->output_0+(elt_2 + 7U)) = ushort(v_3[7U]);
        }
    }
    return;
}

void load_run_0(uint elt_3, array<uint, int(8)> thread* v_4, KernelContext_0 thread* kernelContext_6)
{
    uint _S93 = elt_3 * 2U;
    uint _S94 = as_type<uint>(kernelContext_6->outputBytes_0[(_S93)>>2]);
    uint _S95 = as_type<uint>(kernelContext_6->outputBytes_0[(_S93 + 4U)>>2]);
    uint _S96 = as_type<uint>(kernelContext_6->outputBytes_0[(_S93 + 8U)>>2]);
    uint _S97 = as_type<uint>(kernelContext_6->outputBytes_0[(_S93 + 12U)>>2]);
    (*v_4)[int(0)] = _S94 & 65535U;
    (*v_4)[int(1)] = _S94 >> 16U;
    (*v_4)[int(2)] = _S95 & 65535U;
    (*v_4)[int(3)] = _S95 >> 16U;
    (*v_4)[int(4)] = _S96 & 65535U;
    (*v_4)[int(5)] = _S96 >> 16U;
    (*v_4)[int(6)] = _S97 & 65535U;
    (*v_4)[int(7)] = _S97 >> 16U;
    return;
}

void load_row_0(uint o0_0, uint x0_0, uint width_1, array<uint, int(8)> thread* r_0, KernelContext_0 thread* kernelContext_7)
{
    bool _S98;
    if((x0_0 + 8U) <= width_1)
    {
        _S98 = aligned_run_0(o0_0 + x0_0);
    }
    else
    {
        _S98 = false;
    }
    if(_S98)
    {
        load_run_0(o0_0 + x0_0, r_0, kernelContext_7);
    }
    else
    {
        uint _S99;
        if(x0_0 < width_1)
        {
            _S99 = uint(*(kernelContext_7->output_0+(o0_0 + x0_0)));
        }
        else
        {
            _S99 = 0U;
        }
        (*r_0)[0U] = _S99;
        if((x0_0 + 1U) < width_1)
        {
            _S99 = uint(*(kernelContext_7->output_0+(o0_0 + x0_0 + 1U)));
        }
        else
        {
            _S99 = 0U;
        }
        (*r_0)[1U] = _S99;
        if((x0_0 + 2U) < width_1)
        {
            _S99 = uint(*(kernelContext_7->output_0+(o0_0 + x0_0 + 2U)));
        }
        else
        {
            _S99 = 0U;
        }
        (*r_0)[2U] = _S99;
        if((x0_0 + 3U) < width_1)
        {
            _S99 = uint(*(kernelContext_7->output_0+(o0_0 + x0_0 + 3U)));
        }
        else
        {
            _S99 = 0U;
        }
        (*r_0)[3U] = _S99;
        if((x0_0 + 4U) < width_1)
        {
            _S99 = uint(*(kernelContext_7->output_0+(o0_0 + x0_0 + 4U)));
        }
        else
        {
            _S99 = 0U;
        }
        (*r_0)[4U] = _S99;
        if((x0_0 + 5U) < width_1)
        {
            _S99 = uint(*(kernelContext_7->output_0+(o0_0 + x0_0 + 5U)));
        }
        else
        {
            _S99 = 0U;
        }
        (*r_0)[5U] = _S99;
        if((x0_0 + 6U) < width_1)
        {
            _S99 = uint(*(kernelContext_7->output_0+(o0_0 + x0_0 + 6U)));
        }
        else
        {
            _S99 = 0U;
        }
        (*r_0)[6U] = _S99;
        if((x0_0 + 7U) < width_1)
        {
            _S99 = uint(*(kernelContext_7->output_0+(o0_0 + x0_0 + 7U)));
        }
        else
        {
            _S99 = 0U;
        }
        (*r_0)[7U] = _S99;
    }
    return;
}

void store_row_0(uint o0_1, uint x0_1, uint width_2, const array<uint, int(8)> thread* r_1, KernelContext_0 thread* kernelContext_8)
{
    bool _S100;
    if((x0_1 + 8U) <= width_2)
    {
        _S100 = aligned_run_0(o0_1 + x0_1);
    }
    else
    {
        _S100 = false;
    }
    if(_S100)
    {
        store_run_0(o0_1 + x0_1, r_1, kernelContext_8);
    }
    else
    {
        if(x0_1 < width_2)
        {
            *(kernelContext_8->output_0+(o0_1 + x0_1)) = ushort((*r_1)[0U]);
        }
        if((x0_1 + 1U) < width_2)
        {
            *(kernelContext_8->output_0+(o0_1 + x0_1 + 1U)) = ushort((*r_1)[1U]);
        }
        if((x0_1 + 2U) < width_2)
        {
            *(kernelContext_8->output_0+(o0_1 + x0_1 + 2U)) = ushort((*r_1)[2U]);
        }
        if((x0_1 + 3U) < width_2)
        {
            *(kernelContext_8->output_0+(o0_1 + x0_1 + 3U)) = ushort((*r_1)[3U]);
        }
        if((x0_1 + 4U) < width_2)
        {
            *(kernelContext_8->output_0+(o0_1 + x0_1 + 4U)) = ushort((*r_1)[4U]);
        }
        if((x0_1 + 5U) < width_2)
        {
            *(kernelContext_8->output_0+(o0_1 + x0_1 + 5U)) = ushort((*r_1)[5U]);
        }
        if((x0_1 + 6U) < width_2)
        {
            *(kernelContext_8->output_0+(o0_1 + x0_1 + 6U)) = ushort((*r_1)[6U]);
        }
        if((x0_1 + 7U) < width_2)
        {
            *(kernelContext_8->output_0+(o0_1 + x0_1 + 7U)) = ushort((*r_1)[7U]);
        }
    }
    return;
}

[[kernel]] void scan_planes(uint3 group_1 [[threadgroup_position_in_grid]], uint3 thread_1 [[thread_position_in_threadgroup]], block_task_0 device* blocks_2 [[buffer(1)]], uint32_t device* payload_2 [[buffer(0)]], uint32_t device* outputBytes_2 [[buffer(4)]], ushort device* output_2 [[buffer(3)]], plane_task_0 device* planes_2 [[buffer(2)]], raw_copy_0 device* rawCopies_2 [[buffer(6)]], ushort device* rawValues_2 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_9;
    (&kernelContext_9)->blocks_0 = blocks_2;
    (&kernelContext_9)->payload_0 = payload_2;
    (&kernelContext_9)->outputBytes_0 = outputBytes_2;
    (&kernelContext_9)->output_0 = output_2;
    (&kernelContext_9)->planes_0 = planes_2;
    (&kernelContext_9)->rawCopies_0 = rawCopies_2;
    (&kernelContext_9)->rawValues_0 = rawValues_2;
    uint _S101 = thread_1.x;
    uint t_2 = _S101 % 32U;
    plane_task_0 task_1 = planes_2[group_1.x * 4U + _S101 / 32U];
    bool _S102;
    if((task_1.width_0) > 256U)
    {
        _S102 = true;
    }
    else
    {
        _S102 = (task_1.rows_0) == 0U;
    }
    if(_S102)
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
    load_row_0(task_1.outBase_1, x0_2, task_1.width_0, &next_0, &kernelContext_9);
    uint y_0 = 0U;
    for(;;)
    {
        if(y_0 < (task_1.rows_0))
        {
        }
        else
        {
            break;
        }
        uint o0_2 = task_1.outBase_1 + y_0 * task_1.width_0;
        thread array<uint, int(8)> r_2;
        r_2[0U] = next_0[0U];
        r_2[1U] = next_0[1U];
        r_2[2U] = next_0[2U];
        r_2[3U] = next_0[3U];
        r_2[4U] = next_0[4U];
        r_2[5U] = next_0[5U];
        r_2[6U] = next_0[6U];
        r_2[7U] = next_0[7U];
        uint _S103 = y_0 + 1U;
        if(_S103 < (task_1.rows_0))
        {
            load_row_0(o0_2 + task_1.width_0, x0_2, task_1.width_0, &next_0, &kernelContext_9);
        }
        uint local_0 = r_2[0U] + r_2[1U];
        uint local_1 = local_0 + r_2[2U];
        uint local_2 = local_1 + r_2[3U];
        uint local_3 = local_2 + r_2[4U];
        uint local_4 = local_3 + r_2[5U];
        uint local_5 = local_4 + r_2[6U];
        uint local_6 = local_5 + r_2[7U];
        uint _S104 = simd_prefix_exclusive_sum(local_6);
        uint _S105 = column_0[0U] + (r_2[0U] + _S104);
        column_0[0U] = _S105;
        r_2[0U] = _S105;
        uint _S106 = column_0[1U] + (local_0 + _S104);
        column_0[1U] = _S106;
        r_2[1U] = _S106;
        uint _S107 = column_0[2U] + (local_1 + _S104);
        column_0[2U] = _S107;
        r_2[2U] = _S107;
        uint _S108 = column_0[3U] + (local_2 + _S104);
        column_0[3U] = _S108;
        r_2[3U] = _S108;
        uint _S109 = column_0[4U] + (local_3 + _S104);
        column_0[4U] = _S109;
        r_2[4U] = _S109;
        uint _S110 = column_0[5U] + (local_4 + _S104);
        column_0[5U] = _S110;
        r_2[5U] = _S110;
        uint _S111 = column_0[6U] + (local_5 + _S104);
        column_0[6U] = _S111;
        r_2[6U] = _S111;
        uint _S112 = column_0[7U] + (local_6 + _S104);
        column_0[7U] = _S112;
        r_2[7U] = _S112;
        thread array<uint, int(8)> _S113 = r_2;
        store_row_0(o0_2, x0_2, task_1.width_0, &_S113, &kernelContext_9);
        y_0 = _S103;
    }
    return;
}

[[kernel]] void scan_rows(uint3 group_2 [[threadgroup_position_in_grid]], uint3 thread_2 [[thread_position_in_threadgroup]], block_task_0 device* blocks_3 [[buffer(1)]], uint32_t device* payload_3 [[buffer(0)]], uint32_t device* outputBytes_3 [[buffer(4)]], ushort device* output_3 [[buffer(3)]], plane_task_0 device* planes_3 [[buffer(2)]], raw_copy_0 device* rawCopies_3 [[buffer(6)]], ushort device* rawValues_3 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_10;
    (&kernelContext_10)->blocks_0 = blocks_3;
    (&kernelContext_10)->payload_0 = payload_3;
    (&kernelContext_10)->outputBytes_0 = outputBytes_3;
    (&kernelContext_10)->output_0 = output_3;
    (&kernelContext_10)->planes_0 = planes_3;
    (&kernelContext_10)->rawCopies_0 = rawCopies_3;
    (&kernelContext_10)->rawValues_0 = rawValues_3;
    uint _S114 = thread_2.x;
    uint _S115 = _S114 % 32U;
    plane_task_0 task_2 = planes_3[group_2.x];
    uint y_1 = group_2.y * 4U + _S114 / 32U;
    bool _S116;
    if((task_2.width_0) <= 256U)
    {
        _S116 = true;
    }
    else
    {
        _S116 = y_1 >= (task_2.rows_0);
    }
    if(_S116)
    {
        return;
    }
    uint _S117 = task_2.outBase_1 + y_1 * task_2.width_0;
    uint _S118 = (task_2.width_0 + 256U - 1U) / 256U;
    uint segment_0 = 0U;
    uint rowCarry_0 = 0U;
    for(;;)
    {
        if(segment_0 < _S118)
        {
        }
        else
        {
            break;
        }
        uint xs_0 = segment_0 * 256U + _S115 * 8U;
        thread array<uint, int(8)> r_3;
        load_row_0(_S117, xs_0, task_2.width_0, &r_3, &kernelContext_10);
        uint local_7 = r_3[0U] + r_3[1U];
        uint local_8 = local_7 + r_3[2U];
        uint local_9 = local_8 + r_3[3U];
        uint local_10 = local_9 + r_3[4U];
        uint local_11 = local_10 + r_3[5U];
        uint local_12 = local_11 + r_3[6U];
        uint local_13 = local_12 + r_3[7U];
        uint _S119 = simd_prefix_exclusive_sum(local_13);
        uint _S120 = _S119 + rowCarry_0;
        uint _S121 = simd_sum(local_13);
        uint rowCarry_1 = rowCarry_0 + _S121;
        r_3[0U] = r_3[0U] + _S120;
        r_3[1U] = local_7 + _S120;
        r_3[2U] = local_8 + _S120;
        r_3[3U] = local_9 + _S120;
        r_3[4U] = local_10 + _S120;
        r_3[5U] = local_11 + _S120;
        r_3[6U] = local_12 + _S120;
        r_3[7U] = local_13 + _S120;
        thread array<uint, int(8)> _S122 = r_3;
        store_row_0(_S117, xs_0, task_2.width_0, &_S122, &kernelContext_10);
        segment_0 = segment_0 + 1U;
        rowCarry_0 = rowCarry_1;
    }
    return;
}

[[kernel]] void scan_columns(uint3 group_3 [[threadgroup_position_in_grid]], uint3 thread_3 [[thread_position_in_threadgroup]], block_task_0 device* blocks_4 [[buffer(1)]], uint32_t device* payload_4 [[buffer(0)]], uint32_t device* outputBytes_4 [[buffer(4)]], ushort device* output_4 [[buffer(3)]], plane_task_0 device* planes_4 [[buffer(2)]], raw_copy_0 device* rawCopies_4 [[buffer(6)]], ushort device* rawValues_4 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_11;
    (&kernelContext_11)->blocks_0 = blocks_4;
    (&kernelContext_11)->payload_0 = payload_4;
    (&kernelContext_11)->outputBytes_0 = outputBytes_4;
    (&kernelContext_11)->output_0 = output_4;
    (&kernelContext_11)->planes_0 = planes_4;
    (&kernelContext_11)->rawCopies_0 = rawCopies_4;
    (&kernelContext_11)->rawValues_0 = rawValues_4;
    plane_task_0 task_3 = planes_4[group_3.x];
    uint x_0 = group_3.y * 128U + thread_3.x;
    bool _S123;
    if((task_3.width_0) <= 256U)
    {
        _S123 = true;
    }
    else
    {
        _S123 = x_0 >= (task_3.width_0);
    }
    if(_S123)
    {
        return;
    }
    uint y_2 = 0U;
    uint carry_0 = 0U;
    for(;;)
    {
        if(y_2 < (task_3.rows_0))
        {
        }
        else
        {
            break;
        }
        uint o_0 = task_3.outBase_1 + y_2 * task_3.width_0 + x_0;
        uint carry_1 = carry_0 + uint(*((&kernelContext_11)->output_0+o_0));
        *((&kernelContext_11)->output_0+o_0) = ushort(carry_1);
        y_2 = y_2 + 1U;
        carry_0 = carry_1;
    }
    return;
}

[[kernel]] void copy_raw(uint3 group_4 [[threadgroup_position_in_grid]], uint3 thread_4 [[thread_position_in_threadgroup]], block_task_0 device* blocks_5 [[buffer(1)]], uint32_t device* payload_5 [[buffer(0)]], uint32_t device* outputBytes_5 [[buffer(4)]], ushort device* output_5 [[buffer(3)]], plane_task_0 device* planes_5 [[buffer(2)]], raw_copy_0 device* rawCopies_5 [[buffer(6)]], ushort device* rawValues_5 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_12;
    (&kernelContext_12)->blocks_0 = blocks_5;
    (&kernelContext_12)->payload_0 = payload_5;
    (&kernelContext_12)->outputBytes_0 = outputBytes_5;
    (&kernelContext_12)->output_0 = output_5;
    (&kernelContext_12)->planes_0 = planes_5;
    (&kernelContext_12)->rawCopies_0 = rawCopies_5;
    (&kernelContext_12)->rawValues_0 = rawValues_5;
    raw_copy_0 _S124 = rawCopies_5[group_4.x];
    uint i_1 = thread_4.x;
    for(;;)
    {
        if(i_1 < (_S124.count_1))
        {
        }
        else
        {
            break;
        }
        *((&kernelContext_12)->output_0+(_S124.destination_0 + i_1)) = (&kernelContext_12)->rawValues_0[_S124.source_0 + i_1];
        i_1 = i_1 + 128U;
    }
    return;
}

