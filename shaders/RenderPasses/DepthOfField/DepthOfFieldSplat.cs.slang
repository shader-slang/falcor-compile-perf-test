/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
#include "Utils/NVAPI.slangh" // For fp16 atomics
#include "DepthOfFieldDefines.slangh"

import DepthOfFieldSplat;

/** Compute version of splat pass.
 */
struct SplatPass
{
    // Resources
    ByteAddressBuffer indirectArgs; // Same buffer as used for draw args in raster version
    StructuredBuffer<BokehSplat> bokehBuffer;
    RWTexture2D<float4> outOfFocusRadiance[BOKEH_MIP_LEVELS];

    // Common parameters
    uint2 inputSize;
    float2 invInputSize;
    float cameraAnisotropy;
    float subjectDistance;
    float lensScale;
    float splatComputeMaxRadius;

    struct SplatDesc
    {
        float2 minP; // Top-left corner in range [0,1]
        float2 maxP; // Bottom-right corner in range [0,1]
        uint2 color; // fp16x4
        float radius;
        uint level;
    };

    SplatDesc computeSplat(const uint index)
    {
        const BokehSplat bokeh = bokehBuffer[index];

        uint2 screenPosI = uint2(bokeh.packedScreenPos & 0xFFFF, bokeh.packedScreenPos >> 16);
        float2 screenPos = (float2)screenPosI * .5f;

        // Just scale, 0 means no anisotropy, 1 means inf (it clamped to .95f from the host)
        const float anisoScale = 1.f / (1.f - abs(cameraAnisotropy));
        float2 anisotropyMult = float2(1.f / anisoScale, anisoScale);
        if (cameraAnisotropy < 0.f)
        {
            anisotropyMult = anisotropyMult.yx;
        }

        float radius = bokeh.getRadius();
        uint level = bokeh.getLevel();

        SplatDesc splat;
        splat.minP = mad(-anisotropyMult, abs(radius), screenPos);
        splat.maxP = mad(anisotropyMult, abs(radius), screenPos);
        splat.minP *= invInputSize;
        splat.maxP *= invInputSize;
        splat.color = bokeh.color;
        splat.radius = abs(radius) * (1 << level);
        splat.level = level;

        return splat;
    }

    void drawSplat(SplatDesc splat)
    {
        const float2 outputSize = (float2)(inputSize >> splat.level);
        splat.minP *= outputSize;
        splat.maxP *= outputSize;

        // Compute bounding box of covered pixels (based on pixel centers, inclusive on top-left to match raster tie-breaking rules).
        int2 minP = (int2)max(ceil(splat.minP - 0.5f), float2(0.f));
        int2 maxP = (int2)min(ceil(splat.maxP - 0.5f), outputSize);

        float2 bboxCenter = (splat.minP + splat.maxP) * 0.5f;
        float2 bboxInvSize = float2(2.f) / (splat.maxP - splat.minP);

        // Iterate over pixels in bounding box.
        for (int y = minP.y; y < maxP.y; y++)
        {
            for (int x = minP.x; x < maxP.x; x++)
            {
                float2 pxCenter = float2(x, y) + 0.5f;
                float2 xy = (pxCenter - bboxCenter) * bboxInvSize;

                float distance = length(xy) * splat.radius;
                float fadeStartRadius = max(0.f, splat.radius - max(1.f, splat.radius * 0.2f));
                float intensity = 1.f - smoothstep(fadeStartRadius, splat.radius, distance); // make the circle soft
                if (intensity > 0.f)
                {
                    NvInterlockedAddFp16x4(outOfFocusRadiance[splat.level], uint2(x, y), splat.color);
                }
            }
        }
    }

    /** Entry point for splat pass.
     */
    void execute(const uint threadID)
    {
        // Exit threads that aren't needed.
        const uint splatCount = indirectArgs.Load(INDIRECT_SEPARATION * BOKEH_MIP_LEVELS + 4);
        if (threadID >= splatCount)
            return;

        SplatDesc splat = computeSplat(threadID);
        drawSplat(splat);
    }
};

ParameterBlock<SplatPass> gSplatPass;

[numthreads(256, 1, 1)]
[shader("compute")]
void splatFP16(uint globalId: SV_DispatchThreadID)
{
    gSplatPass.execute(globalId.x);
}
