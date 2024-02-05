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

import Scene.Scene;
import Scene.Camera.Camera;
import Scene.Volume.Grid;
import Scene.Volume.GridVolume;
import Utils.Math.MathHelpers;
import Utils.Geometry.IntersectionHelpers;
import Utils.Sampling.SampleGenerator;
import Rendering.Lights.EnvMapSampler;
import Params;
import PathData;
import WorkQueue;

// Uniform data.
cbuffer PerFrameCB
{
    uint gSeed;
    WorkQueue queue;
    EnvMapSampler gEnvMapSampler;
}

// Inputs.
Texture2D<uint> gAdaptiveSampleMap;

// Per-Sample buffers.
RWStructuredBuffer<ColorData> gSampleColor;
RWTexture2D<uint> gSampleOffsets;

// Shared memory variables.
static const uint kWarpCount = (kScreenTileDim.x * kScreenTileDim.y) / 32;
// TODO: Replace explicitly declared size by compile-time constant when it works. For now assume tile is always 16x16!
// See https://gitlab-master.nvidia.com/nvresearch-gfx/Tools/slang/-/issues/87
groupshared uint gSamplesOffset[8 /* kWarpCount */];
groupshared uint gVolumeOffset[8 /* kWarpCount */];

// --------------------------------------------------------------
// Main entry point.

[numthreads(256 /* kScreenTileDim.x * kScreenTileDim.y */, 1, 1)]
void main(uint3 groupID: SV_GroupID, uint3 groupThreadID: SV_GroupThreadID)
{
    // Compute tile/pixel/store IDs.
    const uint2 tileID = groupID.xy;
    const uint threadID = groupThreadID.x;
    const uint2 tileOffset = tileID << kScreenTileBits;
    const uint2 pixel = deinterleave_8bit(threadID) + tileOffset;

    // Clip against volume AABB.
    float2 nearFar;
    bool hitVolume = false;
    const GridVolume gridVolume = gScene.getGridVolume(0);
    // TODO: AA and DOF for multiple spp?
    const Ray ray = gScene.camera.computeRayPinhole(pixel, kResolution);
    // Note: Do not terminate threads for out-of-bounds pixels because we need all threads active for the prefix sum pass below.
    if (all(pixel < kResolution))
    {
        const AABB bounds = gridVolume.getBounds();
        hitVolume = intersectRayAABB(ray.origin, ray.dir, bounds.minPoint, bounds.maxPoint, nearFar);
    }

    // Adaptive sampling per-tile prefix sum.
    const uint spp =
        all(pixel < kResolution) ? (kUseAdaptiveSampling ? min(gAdaptiveSampleMap[pixel], kMaxSamplesPerPixel) : kSamplesPerPixel) : 0;
    const uint outTileOffset = getTileOffset(tileID);

    // Calculate the path and sample counts over the warp.
    // The first thread in each warp writes the results to shared memory.
    const uint warpID = threadID >> 5;
    {
        uint samples = WaveActiveSum(spp);
        uint volumePaths = WaveActiveSum(hitVolume ? spp : 0);

        if (WaveIsFirstLane())
        {
            gSamplesOffset[warpID] = samples;
            gVolumeOffset[warpID] = volumePaths;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Compute the prefix sum over the warp totals in shared memory.
    // The first N threads in the thread group perform this computation.
    if (threadID < kWarpCount)
    {
        // Compute the prefix sum over the sample counts.
        uint samples = gSamplesOffset[threadID];
        gSamplesOffset[threadID] = WavePrefixSum(samples);

        // Allocate space in the path queues.
        // The return values are the global offsets into the respective queues.
        uint volumePaths = gVolumeOffset[threadID];
        gVolumeOffset[threadID] = queue.addCounter((uint)Counters::kVolumePaths, volumePaths);
    }
    GroupMemoryBarrierWithGroupSync();

    // Terminate out-of-bounds threads.
    if (!all(pixel < kResolution))
        return;

    uint outIdx;
    if (kUseAdaptiveSampling)
    {
        const uint outSampleOffset = gSamplesOffset[warpID] + WavePrefixSum(spp);
        gSampleOffsets[pixel] = outSampleOffset; // For use in later passes.
        outIdx = outTileOffset + outSampleOffset;
    }
    else
    {
        outIdx = outTileOffset + threadID * kSamplesPerPixel;
    }

    // Append paths to queue or write background.
    if (hitVolume)
    {
        const uint idx = gVolumeOffset[warpID] + WavePrefixSum(spp);
        for (uint i = 0; i < spp; ++i)
        {
            // Append to volume queue.
            queue.volumePathIDs.Store((idx + i) * 4, PathState::encodeID(pixel, i));
            queue.volumePathDistances.Store((idx + i) * 4, uint(nearFar.x)); // TODO: remove this, or keep for possible culling?
        }
    }
    else
    {
        // Evaluate background.
        float3 color = float3(0.f);
        if (kShowBackground)
            color = gEnvMapSampler.eval(ray.dir);
        for (uint i = 0; i < spp; ++i)
        {
            // Write output.
            gSampleColor[outIdx + i].set(color);
        }
    }
}
