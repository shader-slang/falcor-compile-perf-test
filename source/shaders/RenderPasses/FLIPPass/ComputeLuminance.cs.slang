/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/

/**
 * Compute shader for computing luminance of an RGB image.
 */

import Utils.Color.ColorHelpers;

cbuffer PerFrameCB
{
    uint2 gResolution;
}

Texture2D<float4> gInputImage;
RWBuffer<float> gOutputLuminance;

[numthreads(16, 16, 1)]
void computeLuminance(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (any(dispatchThreadId.xy >= gResolution))
        return;
    const uint2 pixelPos = dispatchThreadId.xy;

    const float3 color = gInputImage[pixelPos].xyz;

    gOutputLuminance[pixelPos.x + gResolution.x * pixelPos.y] = luminance(color);
}
