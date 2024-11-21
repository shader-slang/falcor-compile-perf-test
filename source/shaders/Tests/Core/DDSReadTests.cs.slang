/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
RWStructuredBuffer<uint> result;
RWStructuredBuffer<float> difference;

Texture2D<float4> tex;
Texture2D<float4> ref;

cbuffer CB
{
    uint2 sz;
}

[numthreads(16, 16, 1)]
void readback(uint3 threadId: SV_DispatchThreadID)
{
    if (any(threadId.xy >= sz))
        return;

    uint idx = threadId.y * sz.x + threadId.x;
    float4 v = tex[threadId.xy];

    // Convert to uint8_t, and pack into uint32_t
    // clang-format off
    result[idx] = ((uint)(saturate(v[3]) * 255) << 24) |
                  ((uint)(saturate(v[2]) * 255) << 16) |
                  ((uint)(saturate(v[1]) * 255) <<  8) |
                  ((uint)(saturate(v[0]) * 255));
    // clang-format on
}

[numthreads(16, 16, 1)]
void diff(uint3 threadId: SV_DispatchThreadID)
{
    if (any(threadId.xy >= sz))
        return;

    uint idx = threadId.y * sz.x + threadId.x;
    float4 tv = tex[threadId.xy];
    float4 rv = tex[threadId.xy];
    for (uint32_t i = 0; i < 4; ++i)
    {
        difference[4 * idx + i] = rv[i] - tv[i];
    }
}
