/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
import Internal.Utils.Math.Float8;

RWStructuredBuffer<uint> result;
StructuredBuffer<float> data;

cbuffer CB
{
    uint numTests;
}

[numthreads(256, 1, 1)]
void testFP8Pack(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    const uint i = dispatchThreadID.x;
    if (i >= numTests)
        return;

    uint d = asuint(data[i]);
    float16_t hval = asfloat16((uint16_t)d);
    bool saturate = d & 0x80000000;

    uint16_t packed = pack_fp8_e4m3(hval, saturate);
    result[i] = (uint)packed;
}

[numthreads(256, 1, 1)]
void testFP8Unpack(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    const uint i = dispatchThreadID.x;
    if (i >= numTests)
        return;

    uint16_t packed = (uint16_t)(i & 0xff);
    float16_t hval = unpack_fp8_e4m3(packed);
    result[i] = (uint)asuint16(hval);
}

void testFP8VectorPackUnpack(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    const uint i = dispatchThreadID.x;

    if (i < numTests / 2)
    {
        float16_t2 hvec = { data[2 * i + 0], data[2 * i + 1] };
        uint16_t packed = pack_fp8_e4m3_t2(hvec, false);
        float16_t2 res = unpack_fp8_e4m3_t2(packed);

        result[2 * i + 0] = (uint)asuint16(res.x);
        result[2 * i + 1] = (uint)asuint16(res.y);
    }

    if (i < numTests / 4)
    {
        float16_t4 hvec = { data[4 * i + 0], data[4 * i + 1], data[4 * i + 2], data[4 * i + 3] };
        uint packed = pack_fp8_e4m3_t4(hvec, false);
        float16_t4 res = unpack_fp8_e4m3_t4(packed);

        result[numTests + 4 * i + 0] = (uint)asuint16(res.x);
        result[numTests + 4 * i + 1] = (uint)asuint16(res.y);
        result[numTests + 4 * i + 2] = (uint)asuint16(res.z);
        result[numTests + 4 * i + 3] = (uint)asuint16(res.w);
    }
}

[numthreads(1, 1, 1)]
void testNaN32(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    // Produce NaNs different ways.
    // Note that the indices must be different, otherwise the compiler statically optimizes it.
    result[0] = asuint(data[0] / data[1]);  // +0/-0
    result[1] = asuint(data[0] / data[2]);  // +0/+0
    result[2] = asuint(data[1] / data[2]);  // -0/+0
    result[3] = asuint(data[1] / data[3]);  // -0/-0
    result[4] = asuint(data[0] * data[4]);  // +0*+inf
    result[5] = asuint(data[0] * data[5]);  // +0*-inf
    result[6] = asuint(data[1] * data[4]);  // -0*+inf
    result[7] = asuint(data[1] * data[5]);  // -0*-inf
    result[8] = asuint(data[6] * data[8]);  // +1*+nan
    result[9] = asuint(data[6] * data[9]);  // +1*-nan
    result[10] = asuint(data[7] * data[8]); // -1*+nan
    result[11] = asuint(data[7] * data[9]); // -1*-nan
}

[numthreads(1, 1, 1)]
void testNaN16(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    // Cast NaNs fp32 -> fp16.
    float16_t a = float16_t(data[0]); // +nan
    float16_t b = float16_t(data[1]); // -nan

    result[0] = (uint)asuint16(a);
    result[1] = (uint)asuint16(b);

    // Cast NaNs fp16 -> fp32.
    float16_t c = asfloat16(uint16_t(asuint(data[2]))); // +nan
    float16_t d = asfloat16(uint16_t(asuint(data[3]))); // -nan

    result[2] = asuint((float)c);
    result[3] = asuint((float)d);
}

[numthreads(256, 1, 1)]
void testRounding16(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    const uint i = dispatchThreadID.x;
    if (i >= numTests)
        return;

    float16_t h = (float16_t)data[i];
    result[i] = (uint)asuint16(h);
}
