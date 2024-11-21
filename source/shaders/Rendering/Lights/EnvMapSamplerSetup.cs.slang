/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/

/** Compute shader for building a hierarchical importance map from an
    environment map. The result is used by EnvMapSampler.slang for sampling.
*/

import Utils.Math.MathHelpers;
import Utils.Color.ColorHelpers;

cbuffer CB
{
    uint2 outputDim;            // Resolution of the importance map in texels.
    uint2 outputDimInSamples;   // Resolution of the importance map in samples.
    uint2 numSamples;           // Per-texel subsamples s.xy at finest mip.
    float invSamples;           // 1 / (s.x*s.y).
};

SamplerState gEnvSampler;
Texture2D<float4> gEnvMap;
RWTexture2D<float> gImportanceMap;

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadID.xy;
    if (any(pixel >= outputDim)) return;

    float L = 0.f;
    for (uint y = 0; y < numSamples.y; y++)
    {
        for (uint x = 0; x < numSamples.x; x++)
        {
            // Compute sample pos p in [0,1)^2 in octahedral map.
            uint2 samplePos = pixel * numSamples + uint2(x, y);
            float2 p = ((float2)samplePos + 0.5f) / outputDimInSamples;

            // Convert p to (u,v) coordinate in latitude-longitude map.
            float3 dir = oct_to_ndir_equal_area_unorm(p);
            float2 uv = world_to_latlong_map(dir);

            // Accumulate the radiance from this sample.
            float3 radiance = gEnvMap.SampleLevel(gEnvSampler, uv, 0).rgb;
            L += luminance(radiance);
        }
    }

    // Store average radiance for this texel.
    gImportanceMap[pixel] = L * invSamples;
}
