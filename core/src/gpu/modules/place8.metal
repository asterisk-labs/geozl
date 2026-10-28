#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct place_copy_0
{
    uint source_0;
    uint sourceRow_0;
    uint sourceStep_0;
    uint destination_0;
    uint destinationRow_0;
    uint destinationStep_0;
    uint width_0;
    uint rows_0;
};

struct KernelContext_0
{
    place_copy_0 device* copies_0;
    packed_uint2 device* target_0;
    packed_uint2 device* source_1;
};

[[kernel]] void place(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_0 [[thread_position_in_threadgroup]], place_copy_0 device* copies_1 [[buffer(1)]], packed_uint2 device* target_1 [[buffer(2)]], packed_uint2 device* source_2 [[buffer(0)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->copies_0 = copies_1;
    (&kernelContext_0)->target_0 = target_1;
    (&kernelContext_0)->source_1 = source_2;
    place_copy_0 c_0 = copies_1[group_0.x];
    uint _S1 = group_0.y * 4U;
    uint _S2 = min(c_0.rows_0, _S1 + 4U);
    uint r_0 = _S1;
    for(;;)
    {
        if(r_0 < _S2)
        {
        }
        else
        {
            break;
        }
        uint _S3 = c_0.source_0 + r_0 * c_0.sourceRow_0;
        uint _S4 = c_0.destination_0 + r_0 * c_0.destinationRow_0;
        uint x_0 = thread_0.x;
        for(;;)
        {
            if(x_0 < (c_0.width_0))
            {
            }
            else
            {
                break;
            }
            *((&kernelContext_0)->target_0+(_S4 + x_0 * c_0.destinationStep_0)) = packed_uint2(uint2(*((&kernelContext_0)->source_1+(_S3 + x_0 * c_0.sourceStep_0))) ) ;
            x_0 = x_0 + 128U;
        }
        r_0 = r_0 + 1U;
    }
    return;
}

