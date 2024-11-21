/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
cbuffer GlobalCB
{
    uint gStride;
}

Buffer<float4> gInBuf;

float4 main(float2 texC: TEXCOORD, float4 posH: SV_POSITION) : SV_Target0
{
    uint2 pixel = uint2(posH.xy);
    uint bufIdx = pixel.x + pixel.y * gStride;
    return gInBuf[bufIdx];
}
