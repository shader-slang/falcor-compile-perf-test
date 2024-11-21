/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"

import Utils.Math.MathHelpers;
import Rendering.Materials.Fresnel;
__exported import Rendering.Materials.IBSDF;

/**
 * Disney's diffuse reflection.
 * Based on https ://blog.selfshadow.com/publications/s2012-shading-course/burley/s2012_pbs_disney_brdf_notes_v3.pdf
 */
struct DisneyDiffuseBRDF : IBSDF, IDifferentiable
{
    float3 albedo;   ///< Diffuse albedo.
    float roughness; ///< Roughness before remapping.

    [Differentiable]
    float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg)
    {
        if (min(wi.z, wo.z) < kMinCosTheta)
            return float3(0.f);

        return evalWeight(wi, wo) * M_1_PI * wo.z;
    }

    bool sample<S : ISampleGenerator>(const float3 wi, out float3 wo, out float pdf, out float3 weight, out uint lobeType, inout S sg)
    {
        wo = sample_cosine_hemisphere_concentric(sampleNext2D(sg), pdf);
        lobeType = (uint)LobeType::DiffuseReflection;

        if (min(wi.z, wo.z) < kMinCosTheta)
        {
            weight = {};
            return false;
        }

        weight = evalWeight(wi, wo);
        return true;
    }

    float evalPdf(const float3 wi, const float3 wo)
    {
        if (min(wi.z, wo.z) < kMinCosTheta)
            return 0.f;

        return M_1_PI * wo.z;
    }

    // private

    // Returns f(wi, wo) * pi.
    [Differentiable]
    float3 evalWeight(float3 wi, float3 wo)
    {
        float3 h = normalize(wi + wo);
        float woDotH = dot(wo, h);
        float fd90 = 0.5f + 2.f * woDotH * woDotH * roughness;
        float fd0 = 1.f;
        float wiScatter = evalFresnelSchlick(fd0, fd90, wi.z);
        float woScatter = evalFresnelSchlick(fd0, fd90, wo.z);
        return albedo * wiScatter * woScatter;
    }

    AlbedoContributions evalAlbedo(const float3 wi, const LobeType lobetype)
    {
        return AlbedoContributions(albedo, 1.0f - albedo, 0.0f, 0.0f);
    }

    RoughnessInformation getRoughnessInformation(const float3 wi)
    {
        RoughnessInformation r;
        r.roughnessBSDFNotation = float2(0.5f, 0.5f);
        return r;
    }
};
