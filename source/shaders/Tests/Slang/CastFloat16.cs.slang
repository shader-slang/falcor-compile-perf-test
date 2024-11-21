/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
RWStructuredBuffer<uint32_t> result;
StructuredBuffer<float16_t> data;

uint32_t pack(float16_t x, float16_t y)
{
    uint16_t ix = asuint16(x);
    uint16_t iy = asuint16(y);
    return (uint32_t(iy) << 16) | ix;
}

[numthreads(16, 1, 1)]
void testCastFloat16(uint3 threadID: SV_DispatchThreadID)
{
    uint i = threadID.x;

    float16_t x = data[2 * i];
    float16_t y = data[2 * i + 1];
    result[i] = pack(x, y);
}
