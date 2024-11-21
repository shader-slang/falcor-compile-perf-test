/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
cbuffer GlobalCB
{
    uint gStride;
}

Texture2D<float4> gInTex;
RWBuffer<float4> gOutBuf;

[numthreads(8, 8, 1)]
void main(uint3 thrdId: SV_DispatchThreadID)
{
    uint bufIdx = thrdId.x + thrdId.y * gStride;
    gOutBuf[bufIdx] = gInTex[thrdId.xy];
}
