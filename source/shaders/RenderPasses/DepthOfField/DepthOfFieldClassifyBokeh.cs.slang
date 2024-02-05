/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
#include "DepthOfFieldDefines.slangh"

import Utils.Math.BitTricks;
import DepthOfFieldSplat;

#ifndef WAVE_OPS_SUPPORTED
#error Expecting WAVE_OPS_SUPPORTED to be defined
#endif

cbuffer gDoFCommonParams
{
    uint2 inputSize;
    float2 invInputSize;
    float cameraAnisotropy;
    float subjectDistance;
    float lensScale;
    float splatComputeMaxRadius; // Max radius for compute-based splat, or -1 if disabled
};

cbuffer gViewParams
{
    float4x4 ClipToView;
};

#if !WAVE_OPS_SUPPORTED
groupshared float4 gWaveEmulation[CLASSIFY_GROUP_SIZE * CLASSIFY_GROUP_SIZE];
#endif

SamplerState gBilinearSampler;

RWByteAddressBuffer gRWExecuteIndirectBuffer;
Texture2D<float4> gInputRadiance;
Texture2D<float> gInputDepth;
RWStructuredBuffer<BokehSplat> RWBokehBuffers[BOKEH_MIP_LEVELS + 1];

uint2 float4ToPackedHalf4(float4 unpackedInput)
{
    uint2 packedOutput;
    packedOutput.x = f32tof16(unpackedInput.x) | (f32tof16(unpackedInput.y) << 16);
    packedOutput.y = f32tof16(unpackedInput.z) | (f32tof16(unpackedInput.w) << 16);
    return packedOutput;
}

// computeCoCRadius: returns the radius (in pixels) of the circle of confusion for a point at the given depth
//
// computeCoCRadius multiplies the per-pixel variable factor by the constant factor
// in the 'CoC Diameter' formula derived from the Gaussian lens equation in
// https://en.wikipedia.org/wiki/Circle_of_confusion#Determining_a_circle_of_confusion_diameter_from_the_object_field
//   c={|S_{2}-S_{1}| \over S_{2}}{f^{2} \over N(S_{1}-f)}\,
//
// I.e.: we are doing here the '(S2 - S1)/S2' part
//  S2 = depth (e.g.: at a given pixel)
//  S1 = subjectDistance (i.e.: the distance to the point/subject in focus)
//
// Note there are two extra factors folded into lensScale in DoF.cpp: filmPlaneToPixels and diameterToRadius
//
float computeCoCRadius(float depth)
{
    return clamp(lensScale * (depth - subjectDistance) / depth, -LARGEST_RADIUS, LARGEST_RADIUS);
}

float4 computeCoCRadius(float4 depth)
{
    return float4(computeCoCRadius(depth.x), computeCoCRadius(depth.y), computeCoCRadius(depth.z), computeCoCRadius(depth.w));
}

float linearizeDepth(float depth)
{
    float4 clipPos = float4(0.f, 0.f, depth, 1.f);
    float4 viewPos = mul(ClipToView, clipPos);
    return abs(viewPos.z / viewPos.w);
}

// Splats in higher mip levels are slightly expanded by the upsampling filter.
// Compensate for that by reducing their radius.
float adjustRadiusForLevel(float radius, uint level)
{
    // The power function base is determined experimentally to provide the most stable image.
    return (abs(radius) - pow(1.8f, level) + 1.f) * sign(radius);
}

float getSplatWeight(float radius, float adjustedRadius, float lensScale)
{
    float f = clamp((lensScale - radius) / (LARGEST_RADIUS + lensScale), 0.1f, 1.f);
    f *= f;
    adjustedRadius = max(abs(adjustedRadius), SMALLEST_RADIUS);
    f *= 1.f / ((float)M_PI * adjustedRadius * adjustedRadius);
    return f;
}

void appendSplat(float signedRadius, float radiusBias, uint2 bokehPixel, float3 radiance, float weightScale)
{
    if (abs(signedRadius) < SMALLEST_RADIUS)
        return;

    if (any((bokehPixel >> 1) > inputSize))
        return;

    float bokehRadius = abs(signedRadius) + radiusBias;
    uint level = clamp(ceil(log2(bokehRadius / MAX_RADIUS)), 0, BOKEH_MIP_LEVELS - 1);
    bokehRadius *= sign(signedRadius);
    float adjustedRadius = adjustRadiusForLevel(bokehRadius, level);
    float weight = weightScale * getSplatWeight(bokehRadius, adjustedRadius, lensScale);

    BokehSplat bokeh;
    bokeh.packedScreenPos = (uint)((bokehPixel.y << 16) | bokehPixel.x);
    bokeh.packedRadiusLevel = (f32tof16(adjustedRadius) << 16) | level;
    bokeh.color = float4ToPackedHalf4(float4(radiance, 1.f) * weight);

    // If compute-based splatting is enabled, redirect all splats under the radius threshold
    // to the compute queue (last splat buffer).
    if (splatComputeMaxRadius > 0.f && abs(adjustedRadius) <= splatComputeMaxRadius)
        level = BOKEH_MIP_LEVELS;

    uint idx = 0;
#if WAVE_OPS_SUPPORTED
    if (WaveActiveAllEqual(level))
    {
        // If all lanes in the warp are writing their splats into the same list, we can optimize
        // the atomic operation (InterlockedAdd) by counting the active lanes and doing a single
        // atomic for all of them at once, then distributing results among the lanes.
        // The implementation could use fewer Wave functions if it relied on masks, but the MSFT
        // reference on these functions recommends avoiding masks to get better support for
        // machines with different wave sizes, i.e. higher than 32.

        uint currentLane = WaveGetLaneIndex();

#if 0
        // Causes a compile error in Slang/Glslang for Vulkan
        uint activeCount = WaveActiveCountBits(true);
#else
        uint4 activeLanes = WaveActiveBallot(true);
        uint activeCount = countbits(activeLanes.x) + countbits(activeLanes.y);
#endif

        uint firstActive = WaveReadLaneFirst(currentLane);

        if (currentLane == firstActive)
        {
            gRWExecuteIndirectBuffer.InterlockedAdd(level * INDIRECT_SEPARATION + 4, activeCount, idx);
        }

        idx = WaveReadLaneAt(idx, firstActive);
        idx += WavePrefixCountBits(true);
    }
    else
#endif
    {
        gRWExecuteIndirectBuffer.InterlockedAdd(level * INDIRECT_SEPARATION + 4, 1, idx);
    }

    RWBokehBuffers[NonUniformResourceIndex(level)][idx] = bokeh;
}

float minMagnitude(float2 v)
{
    float2 s = sign(v);
    v = abs(v);
    bool sameSign = s.x == s.y;
    return sameSign ? min(v.x, v.y) * s.x : 0.f;
}

float minMagnitude(float4 v)
{
    float4 s = sign(v);
    v = abs(v);
    bool sameSign = s.x == s.y && s.x == s.z && s.x == s.w;
    return sameSign ? min(min(v.x, v.y), min(v.z, v.w)) * s.x : 0.f;
}

// Combines the current thread's color and radius with ones from an adjacent thread,
// determined by (laneIndex ^ offset). With the Z-curve ordering of threads,
// power-of-2 lane offsets can mean either a horizontal or vertical pixel offset.
void reduceColorAndRadius(inout float3 color, inout float radius, int offset, uint threadId)
{
#if WAVE_OPS_SUPPORTED
    uint otherLane = WaveGetLaneIndex() ^ offset;
    float4 data = float4(color, radius);
    float4 otherData = WaveReadLaneAt(data, otherLane);
#else
    float4 otherData = gWaveEmulation[threadId ^ offset];
#endif
    color = (color + otherData.rgb) * 0.5f;
    radius = minMagnitude(float2(radius, otherData.w));
}

/** Classify bokeh pass.
    The thread groups are 256x1 mapped to 16x16 tiles in Z-order. Each thread operates on 2x2 pixels.
*/
[numthreads(CLASSIFY_GROUP_SIZE * CLASSIFY_GROUP_SIZE, 1, 1)]
[shader("compute")]
void classifyBokeh(uint2 groupIndex: SV_GroupID, uint2 threadInGroup: SV_GroupThreadID)
{
    // Rename the linear thread group into a 2D grid so that we can compute
    // a spatial reduction with wave intrinsics
    uint2 pixelIndex = deinterleave_16bit(threadInGroup.x);
    pixelIndex += groupIndex * CLASSIFY_GROUP_SIZE;

    uint2 bottomRight = uint2(inputSize) - 1;

    float2 tc = (2 * pixelIndex + 1) * invInputSize;
    float4 depth = gInputDepth.GatherRed(gBilinearSampler, tc);

    depth.w = linearizeDepth(depth.w);
    depth.z = linearizeDepth(depth.z);
    depth.y = linearizeDepth(depth.y);
    depth.x = linearizeDepth(depth.x);

    float4 radius = computeCoCRadius(depth);

    float3 fullResColors[4];
    fullResColors[3] = gInputRadiance[min(2 * pixelIndex, bottomRight)].rgb;
    fullResColors[2] = gInputRadiance[min(2 * pixelIndex + uint2(1, 0), bottomRight)].rgb;
    fullResColors[1] = gInputRadiance[min(2 * pixelIndex + uint2(1, 1), bottomRight)].rgb;
    fullResColors[0] = gInputRadiance[min(2 * pixelIndex + uint2(0, 1), bottomRight)].rgb;

    float halfResRadius = minMagnitude(radius);
    float3 halfResColor = (fullResColors[0] + fullResColors[1] + fullResColors[2] + fullResColors[3]) * 0.25f;

#if !WAVE_OPS_SUPPORTED
    gWaveEmulation[threadInGroup.x] = float4(halfResColor, halfResRadius);
    GroupMemoryBarrierWithGroupSync();
#endif

    float quarterResRadius = halfResRadius;
    float3 quarterResColor = halfResColor;
    reduceColorAndRadius(quarterResColor, quarterResRadius, 1, threadInGroup.x); // horizontal
    reduceColorAndRadius(quarterResColor, quarterResRadius, 2, threadInGroup.x); // vertical

#if !WAVE_OPS_SUPPORTED
    reduceColorAndRadius(quarterResColor, quarterResRadius, 3, threadInGroup.x); // diagonal

    GroupMemoryBarrierWithGroupSync();
    gWaveEmulation[threadInGroup.x] = float4(quarterResColor, quarterResRadius);
    GroupMemoryBarrierWithGroupSync();
#endif

    float eighthResRadius = quarterResRadius;
    float3 eighthResColor = quarterResColor;
    reduceColorAndRadius(eighthResColor, eighthResRadius, 4, threadInGroup.x); // horizontal
    reduceColorAndRadius(eighthResColor, eighthResRadius, 8, threadInGroup.x); // vertical

#if !WAVE_OPS_SUPPORTED
    reduceColorAndRadius(eighthResColor, eighthResRadius, 4 | 8, threadInGroup.x); // diagonal
#endif

    // TODO (Max): currently the limit on the resolution is 8k x 8k because of how pixel indexing is handled
    // this can be improved to 16k x 16k with minor modifications
    if (abs(eighthResRadius) > EIGHTH_RES_THRESHOLD)
    {
        if ((threadInGroup.x & 0xf) == 0)
        {
            // bokeh per 8x8
            appendSplat(eighthResRadius, 7.f / sqrt(2.f), pixelIndex * 4 + 8, eighthResColor, 64);
        }
    }
    else if (abs(quarterResRadius) > QUARTER_RES_THRESHOLD)
    {
        if ((threadInGroup.x & 0x3) == 0)
        {
            // bokeh per 4x4
            appendSplat(quarterResRadius, 3.f / sqrt(2.f), pixelIndex * 4 + 4, quarterResColor, 16);
        }
    }
    else if (abs(halfResRadius) > HALF_RES_THRESHOLD)
    {
        // bokeh per 2x2
        appendSplat(halfResRadius, 1.f / sqrt(2.f), pixelIndex * 4 + 2, halfResColor, 4);
    }
    else
    {
        // bokeh per pixel, write out 4
        appendSplat(radius.w, 0, pixelIndex * 4 + uint2(1, 1), fullResColors[3], 1);
        appendSplat(radius.z, 0, pixelIndex * 4 + uint2(3, 1), fullResColors[2], 1);
        appendSplat(radius.y, 0, pixelIndex * 4 + uint2(3, 3), fullResColors[1], 1);
        appendSplat(radius.x, 0, pixelIndex * 4 + uint2(1, 3), fullResColors[0], 1);
    }
}
