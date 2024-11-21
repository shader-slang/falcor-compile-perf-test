/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"

import Scene.Shading;
import Utils.Math.MathHelpers;
import Utils.Sampling.UniformSampleGenerator;

/** Utility for integrating an isotropic BSDF over the hemisphere.

    The code assumes thread group size 1024 and wave size 32.
    The execution is arranged in groups of 32x32 threads over the 2D integration grid.
    The third dimension is the grid index (= which incident direction to use).
    The result is reduced to one value per thread group before writing out.
*/
struct BSDFIntegrator
{
    uint2 gridSize;                             ///< Size of integration grid over the hemisphere.
    uint gridCount;                             ///< Integration grid count. One incident direction per grid.
    uint resultCount;                           ///< Number of intermediate results per grid.
    uint materialID;                            ///< Material ID in the scene.
    StructuredBuffer<float> cosThetas;          ///< Cosine theta of incident directions.
    RWStructuredBuffer<float3> results;         ///< Intermediate results. One value per thread group.
    RWStructuredBuffer<float3> finalResults;    ///< Final results. One value per grid.

    static const uint2 kSampleCount = { 8, 8 }; ///< Number of stratified samples per grid cell.

    static const uint kWaveSize = 32;
    static groupshared float3 sharedResult[1024 / kWaveSize]; // Note Slang requires static when declaring groupshared inside struct.

    /** Integrate BSDF over the hemisphere.
        The function is executed with one thread per grid cell, and the result reduced to a single value per thread group.
    */
    void executeIntegration(const uint2 threadID, const uint groupThreadIdx, const uint3 groupID)
    {
        if (groupID.z >= gridCount) return;

        // Setup shading data struct.
        ShadingData sd = {};
        sd.frame = ShadingFrame::createIdentity();
        sd.faceN = sd.frame.N;
        sd.frontFacing = true;

        sd.mtl = gScene.materials.getMaterialHeader(materialID);
        sd.materialID = materialID;
        sd.IoR = 1.f;

        float cosTheta = saturate(cosThetas[groupID.z]);
        float sinTheta = sqrt(1.f - cosTheta * cosTheta);
        sd.V = float3(sinTheta, 0.f, cosTheta);

        // Create MaterialInstance.
        let lod = ExplicitLodTextureSampler(0.f);
        let mi = gScene.materials.getMaterialInstance(sd, lod);

        // Approximate integral of the BSDF over grid cell.
        // We take multiple stratified samples to reduce the integration error.
        // The samples are currently placed at the center of each strata.
        float3 value = {};

        UniformSampleGenerator sg = UniformSampleGenerator(threadID, 0);
        for (uint sy = 0; sy < kSampleCount.y; sy++)
        {
            for (uint sx = 0; sx < kSampleCount.x; sx++)
            {
                // Compute sample position in [0,1]^2 and map to hemisphere.
                float2 offset = (float2(sx, sy) + 0.5f) * (1.f / kSampleCount);
                float2 u = (float2(threadID) + offset) / gridSize;

                float pdf = 0.f;
                float3 wo = sample_cosine_hemisphere_concentric(u, pdf);

                // Evaluate BSDF divided by pdf.
                value += pdf > 0.f ? mi.eval(sd, wo, sg) / pdf : float3(0.f);
            }
        }
        value *= 1.f / (kSampleCount.x * kSampleCount.y);

        // Reduce value over thread group.
        reduceGroup(value, groupThreadIdx, groupID);
    }

    void reduceWarpToShared(float3 value, const uint groupThreadIdx)
    {
        // Accumulate over warp. Store result to shared memory.
        value = WaveActiveSum(value);
        if (WaveIsFirstLane())
        {
            sharedResult[groupThreadIdx / kWaveSize] = value;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    /** Reduce result over thread group and write to memory.
    */
    void reduceGroup(float3 value, const uint groupThreadIdx, const uint3 groupID, const uint kGroupSize = 1024)
    {
        reduceWarpToShared(value, groupThreadIdx);

        // First warp in group accumulates values from shared memory.
        if (groupThreadIdx < (kGroupSize / kWaveSize))
        {
            value = sharedResult[groupThreadIdx];
            value = WaveActiveSum(value);

            // First thread in group stores result to global memory.
            if (groupThreadIdx == 0)
            {
                uint groupIdx = (gridSize.x / 32 /* group width */) * groupID.y + groupID.x;
                uint dstIdx = resultCount * groupID.z + groupIdx;

                results[dstIdx] = value * (1.f / kGroupSize);
            }
        }
    }

    /** Reduce results buffer to a single final value per grid.
    */
    void executeFinal(const uint groupThreadIdx, const uint gridIdx, const uint kGroupSize = 256)
    {
        if (gridIdx >= gridCount) return;

        uint srcIdx = resultCount * gridIdx + groupThreadIdx;
        float3 value = results[srcIdx];

        reduceWarpToShared(value, groupThreadIdx);

        // First warp in group accumulates values from shared memory.
        if (groupThreadIdx < (kGroupSize / kWaveSize))
        {
            value = sharedResult[groupThreadIdx];
            value = WaveActiveSum(value);

            // First thread in group stores result to global memory.
            if (groupThreadIdx == 0)
            {
                finalResults[gridIdx] = value * (1.f / kGroupSize);
            }
        }
    }
};

ParameterBlock<BSDFIntegrator> gIntegrator;

// TODO: Specify required wave size when Slang supports it. See https://gitlab-master.nvidia.com/nvresearch-gfx/Tools/Falcor/-/issues/1414
//[WaveSize(32)]
[numthreads(32, 32, 1)]
void mainIntegration(uint3 threadID : SV_DispatchThreadID, uint groupThreadIdx : SV_GroupIndex, uint3 groupID : SV_GroupID)
{
    gIntegrator.executeIntegration(threadID.xy, groupThreadIdx, groupID);
}

// TODO: Specify required wave size when Slang supports it. See https://gitlab-master.nvidia.com/nvresearch-gfx/Tools/Falcor/-/issues/1414
//[WaveSize(32)]
[numthreads(256, 1, 1)]
void mainFinal(uint3 threadID : SV_DispatchThreadID)
{
    gIntegrator.executeFinal(threadID.x, threadID.y);
}
