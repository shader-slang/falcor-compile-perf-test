/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
RWStructuredBuffer<uint> result;

#if FLOAT_FORMAT
Texture2D<float4> tex;
#else
Texture2D<uint4> tex;
#endif

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
#if FLOAT_FORMAT
    float4 v = tex[threadId.xy];
#else
    uint4 v = tex[threadId.xy];
#endif

    for (uint i = 0; i < 4; i++)
    {
        result[idx * 4 + i] = asuint(v[i]);
    }
}
