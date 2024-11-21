/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/

// This is different than the ConvertTexToBuf.cs.slang because Falcor's motion
//    vectors are in a different format than expected by OptiX.  This shader
//    also converts formats.

cbuffer GlobalCB
{
    uint gStride;
    uint2 gSize;
}

Texture2D<float4> gInTex;
RWBuffer<float2> gOutBuf;

[numthreads(8, 8, 1)]
void main(uint3 thrdId: SV_DispatchThreadID)
{
    uint bufIdx = thrdId.x + thrdId.y * gStride;
    float2 optixMVec = gInTex[thrdId.xy].xy * gSize;
    gOutBuf[bufIdx] = float2(-optixMVec.x, -optixMVec.y);
}
