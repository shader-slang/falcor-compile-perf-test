/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
RWStructuredBuffer<uint64_t> result;

#if USE_UAV
RWStructuredBuffer<float64_t> data;
#else
StructuredBuffer<float64_t> data;
#endif

[numthreads(16, 1, 1)]
void testFloat64(uint3 threadID: SV_DispatchThreadID)
{
    uint i = threadID.x;

    float64_t value = data[i]; // 64-bit load

    uint2 tmp;
    asuint(value, tmp.x, tmp.y);
    result[i] = uint64_t(tmp.x) | uint64_t(tmp.y) << 32;
}
