/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"

import Scene.Scene;
import Utils.Color.ColorHelpers;

/** Update the env light luminance and PDF textures by sampling the scene's env map.
*/
struct EnvLightUpdater
{
    RWTexture2D<float> envLightLuminance;
    RWTexture2D<float> envLightPdf;

    uint2 texDim;

    void execute(uint2 texelID)
    {
        if (any(texelID >= texDim)) return;

        // Compute UV coordinates in env map.
        float2 uv = (texelID + float2(0.5f)) / texDim;

        // Evaluate the env map luminance.
        float L = max(0.f, luminance(gScene.envMap.eval(uv)));

        // Write luminance.
        envLightLuminance[texelID] = L;

        // Compute relative solid angle to account for compression at the poles.
        float elevation = (((float(texelID.y) + 0.5f) / float(texDim.y)) - 0.5f) * M_PI;
        float relativeSolidAngle = cos(elevation);

        // Write PDF.
        envLightPdf[texelID] = relativeSolidAngle * L;
    }
};

ParameterBlock<EnvLightUpdater> gEnvLightUpdater;

[numthreads(16, 16, 1)]
void main(uint2 threadID : SV_DispatchThreadID)
{
    gEnvLightUpdater.execute(threadID);
}
