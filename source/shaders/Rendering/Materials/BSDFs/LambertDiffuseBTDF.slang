/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"

import Utils.Math.MathHelpers;
__exported import Rendering.Materials.IBSDF;

/**
 * Lambertian diffuse transmission.
 */
struct LambertDiffuseBTDF : IBSDF, IDifferentiable
{
    float3 albedo; ///< Diffuse albedo.

    [Differentiable]
    float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg)
    {
        if (min(wi.z, -wo.z) < kMinCosTheta)
            return float3(0.f);

        return M_1_PI * albedo * -wo.z;
    }

    bool sample<S : ISampleGenerator>(const float3 wi, out float3 wo, out float pdf, out float3 weight, out uint lobeType, inout S sg)
    {
        wo = sample_cosine_hemisphere_concentric(sampleNext2D(sg), pdf);
        wo.z = -wo.z;
        lobeType = (uint)LobeType::DiffuseTransmission;

        if (min(wi.z, -wo.z) < kMinCosTheta)
        {
            weight = {};
            return false;
        }

        weight = albedo;
        return true;
    }

    float evalPdf(const float3 wi, const float3 wo)
    {
        if (min(wi.z, -wo.z) < kMinCosTheta)
            return 0.f;

        return M_1_PI * -wo.z;
    }

    AlbedoContributions evalAlbedo(const float3 wi, const LobeType lobetype)
    {
        return AlbedoContributions(0.0f, 0.0f, albedo, 1.0f - albedo);
    }

    RoughnessInformation getRoughnessInformation(const float3 wi)
    {
        RoughnessInformation r;
        r.roughnessBSDFNotation = float2(0.5f, 0.5f);
        return r;
    }
}
