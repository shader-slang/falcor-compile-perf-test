/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"

__exported import Rendering.Materials.IMaterialInstance;
import Utils.Color.ColorHelpers;
import Utils.Math.MathHelpers;

/** PBRT diffuse transmisson material instance.
    Note that the BSDF is implemented to handle wi/wo over the full sphere.
    The material is treated as single-sided and the shading normal should not be flipped.
*/
struct PBRTDiffuseTransmissionMaterialInstance : MaterialInstanceBase, IMaterialInstance
{
    ShadingFrame sf;            ///< Shading frame in world space.
    float3 albedoR;
    float3 albedoT;

    float3 eval<S : ISampleGenerator>(const ShadingData sd, const float3 wo, inout S sg)
    {
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = sf.toLocal(wo);

        if (!isValidHemisphereReflectionOrTransmission(sd, sf, wiLocal, woLocal, wo, true)) return float3(0.f);

        float3 albedo = (wiLocal.z * woLocal.z > 0.0f) ? albedoR : albedoT;
        return albedo * abs(woLocal.z) * M_1_PI;
    }

    [Differentiable]
    float3 evalAD<S : ISampleGenerator>(const DiffMaterialData diffData, const ShadingData sd, const float3 wo, inout S sg)
    {
        return float3(0.f);
    }

    bool sample<S : ISampleGenerator>(const ShadingData sd, inout S sg, out BSDFSample result, bool useImportanceSampling = true)
    {
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = sample_cosine_hemisphere_concentric(sampleNext2D(sg), result.pdf);
        if (wiLocal.z < 0.0f) woLocal = -woLocal;

        float PR = luminance(albedoR);
        float PT = luminance(albedoT);
        if (PR + PT == 0.0f) return false;
        if (sampleNext1D(sg) < PT / (PR + PT))
        {
            woLocal = -woLocal;
            result.weight = albedoT / PT * (PR + PT);
            result.lobeType = (uint)LobeType::DiffuseTransmission;
            result.pdf *= PT / (PR + PT);
        }
        else
        {
            result.weight = albedoR / PR * (PR + PT);
            result.lobeType = (uint)LobeType::DiffuseReflection;
            result.pdf *= PR / (PR + PT);
        }
        result.wo = sf.fromLocal(woLocal);

        if (!isValidHemisphereReflectionOrTransmission(sd, sf, wiLocal, woLocal, result.wo, true)) return false;

        return true;
    }

    float evalPdf(const ShadingData sd, const float3 wo, bool useImportanceSampling = true)
    {
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = sf.toLocal(wo);

        if (!isValidHemisphereReflectionOrTransmission(sd, sf, wiLocal, woLocal, wo, true)) return 0.f;

        float PR = luminance(albedoR);
        float PT = luminance(albedoT);
        if (PR + PT == 0.0f) return 0.0f;
        float P = (wiLocal.z * woLocal.z > 0.0f) ? PR / (PR + PT) : PT / (PR + PT);

        return P * abs(woLocal.z) * M_1_PI;
    }

    BSDFProperties getProperties(const ShadingData sd)
    {
        BSDFProperties p = {};

        p.guideNormal = sf.N;
        p.roughness = 1.0f;
        p.diffuseReflectionAlbedo = albedoR;
        p.diffuseTransmissionAlbedo = albedoT;
        p.flags |= (uint)BSDFProperties::Flags::IsTransmissive;

        return p;
    }

    uint getLobeTypes(const ShadingData sd)
    {
        return (uint)LobeType::DiffuseReflection | (uint)LobeType::DiffuseTransmission;
    }
};
