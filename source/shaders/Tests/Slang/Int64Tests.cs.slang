/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
RWStructuredBuffer<uint> result;

#if USE_UAV
RWStructuredBuffer<uint64_t> data;
#else
StructuredBuffer<uint64_t> data;
#endif

[numthreads(16, 1, 1)]
void testInt64(uint3 threadID: SV_DispatchThreadID)
{
    uint i = threadID.x;

    uint64_t value = data[i]; // 64-bit load

    uint lo = (uint)value;
    uint hi = (uint)(value >> 32);
    result[2 * i] = lo;
    result[2 * i + 1] = hi;
}
