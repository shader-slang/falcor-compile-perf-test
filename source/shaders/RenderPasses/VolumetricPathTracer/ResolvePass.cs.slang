/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"

import Utils.Math.PackedFormats;
import Utils.Math.FormatConversion;

import Params;
import PathData;

// Inputs.
Texture2D<uint> gAdaptiveSampleMap;

// Outputs.
RWTexture2D<float4> gOutputColorTexture;
StructuredBuffer<ColorData> gSampleColor;
Texture2D<uint> gSampleOffsets;

// --------------------------------------------------------------
// Main entry point.

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    const uint2 pixel = dispatchThreadId.xy;
    if (any(pixel >= kResolution))
        return;
    const uint offset = getSampleOffset(pixel, gSampleOffsets);
    const uint spp = kUseAdaptiveSampling ? min(gAdaptiveSampleMap[pixel], kMaxSamplesPerPixel) : kSamplesPerPixel;

    // Resolve per-sample buffers.
    float3 color = float3(0.f);
    for (uint i = 0; i < spp; ++i)
    {
        const uint idx = offset + i;
        const float3 sampleColor = gSampleColor[idx].get();
        color += sampleColor;
    }

    // Normalize and write results.
    const float invSPP = 1.f / max(1.f, spp);
    gOutputColorTexture[pixel] = float4(color * invSPP, 1.f);
}
