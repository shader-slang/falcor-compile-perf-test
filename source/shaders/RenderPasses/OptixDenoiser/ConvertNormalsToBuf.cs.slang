/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
cbuffer GlobalCB
{
    uint gStride;
    float4x4 gViewIT;
}

Texture2D<float4> gInTex;
RWBuffer<float3> gOutBuf;

[numthreads(8, 8, 1)]
void main(uint3 thrdId: SV_DispatchThreadID)
{
    uint bufIdx = thrdId.x + thrdId.y * gStride;

    float3 normal = (gInTex[thrdId.xy].xyz - 0.5f) * 2.0f;
    if (length(normal) > 0.01)
    {
        normal = mul(gViewIT, float4(normal, 0.0f)).xyz;
        normal = normalize(normal);
    }

    gOutBuf[bufIdx] = normal;
}
