/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"

struct LightProfile
{
    float fluxFactor;

    Texture2D texture;
    SamplerState sampler;

    float getFluxFactor() { return fluxFactor; }

    float eval(float cosTheta)
    {
        float theta = acos(saturate(cosTheta));
        float normTheta = theta / M_PI;

        return texture.SampleLevel(sampler, float2(normTheta, 0.f), 0).x;
    }
};
