/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
cbuffer CB
{
    uint2 frameDim;
    float scaleA;
    float scaleB;
}

// Inputs
Texture2D<float4> A;
Texture2D<float4> B;

// Output
RWTexture2D<float4> output;

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    const uint2 pixel = dispatchThreadId.xy;
    if (any(pixel >= frameDim))
        return;

    float4 result = (scaleA * A[pixel]) + (scaleB * B[pixel]);

    output[pixel] = result;
}
