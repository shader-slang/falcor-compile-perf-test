/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"
#include "rtxdi/RtxdiMath.hlsli"

import Scene.Scene;
import Utils.Color.ColorHelpers;
import PolymorphicLight;

/** Update the light info data used by RTXDI by reading from the scene data.
*/
struct LightUpdater
{
    RWStructuredBuffer<PackedPolymorphicLight> lightInfo;   ///< RTXDI light infos.
    RWTexture2D<float> localLightPdf;                       ///< RTXDI local light pdf texture.

    StructuredBuffer<uint> analyticLightIDs;                ///< Mapping from [0..N) to analytic light IDs in the scene.

    uint2 threadCount;                                      ///< Launch dimensions.

    uint totalLightCount;                                   ///< Total number of lights.
    uint firstLocalAnalyticLight;                           ///< Index of first local analytic light.
    uint firstInfiniteAnalyticLight;                        ///< Index of first infinite analytic light.
    uint envLightIndex;                                     ///< Index of the env map.

    bool updateEmissiveLights;                              ///< True if emissive lights should be updated.
    bool updateEmissiveLightsFlux;                          ///< True if emissive lights flux should be updated.

    bool updateAnalyticLights;                              ///< True if analytic lights should be updated.
    bool updateAnalyticLightsFlux;                          ///< True if analytic lights flux should be updated.

    /** Update the light info and local light PDF texture.
    */
    void execute(uint2 threadID)
    {
        const uint lightIdx = threadID.y * threadCount.x + threadID.x;
        if (lightIdx >= totalLightCount) return;

        if (lightIdx < firstLocalAnalyticLight)
        {
            // Create emissive lights.

            // Get the scene triangle index of the emissive light.
            const uint triIdx = gScene.lightCollection.activeTriangles[lightIdx];

            // Load emissive triangle data.
            const EmissiveTriangle et = gScene.lightCollection.getTriangle(triIdx);
            const EmissiveFlux ef = gScene.lightCollection.fluxData[triIdx];

            // Setup emissive light.
            EmissiveLight light = {};
            light.triIdx = triIdx;
            light.position = et.posW[0];
            light.edge0 = et.posW[1] - et.posW[0];
            light.edge1 = et.posW[2] - et.posW[0];
            light.averageRadiance = luminance(ef.averageRadiance);

            if (updateEmissiveLights || updateEmissiveLightsFlux)
            {
                lightInfo[lightIdx] = light.pack();
            }

            if (updateEmissiveLightsFlux)
            {
                localLightPdf[RTXDI_LinearIndexToZCurve(lightIdx)] = light.getFlux();
            }
        }
        else if (lightIdx < envLightIndex)
        {
            // Create analytic lights.

            // Get the index of the light in the scene.
            uint lightID = analyticLightIDs[lightIdx - firstLocalAnalyticLight];

            // Setup analytic light.
            AnalyticLight light = {};
            light.lightID = lightID;

            if (updateAnalyticLights || updateAnalyticLightsFlux)
            {
                lightInfo[lightIdx] = light.pack();
            }

            if (updateAnalyticLightsFlux && lightIdx < firstInfiniteAnalyticLight)
            {
                localLightPdf[RTXDI_LinearIndexToZCurve(lightIdx)] = light.getFlux();
            }
        }
        else // if (lightIdx == envLightIndex)
        {
            // Create env light.

            EnvLight light = {};
            lightInfo[lightIdx] = light.pack();
        }
    }
};

ParameterBlock<LightUpdater> gLightUpdater;

[numthreads(16, 16, 1)]
void main(uint2 threadID : SV_DispatchThreadID)
{
    gLightUpdater.execute(threadID);
}
