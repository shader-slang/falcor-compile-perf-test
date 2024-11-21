/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
RWStructuredBuffer<uint16_t> result;

#if USE_UAV
RWStructuredBuffer<float16_t> data;
#else
StructuredBuffer<float16_t> data;
#endif

[numthreads(16, 1, 1)]
void testFloat16(uint3 threadID: SV_DispatchThreadID)
{
    uint i = threadID.x;

    float16_t value = data[i]; // 16-bit load

    result[i] = f32tof16(value);
}
