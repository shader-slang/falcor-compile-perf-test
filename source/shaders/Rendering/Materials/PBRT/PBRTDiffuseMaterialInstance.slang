/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"

__exported import Rendering.Materials.IMaterialInstance;
__exported import DiffRendering.SceneGradients;
__exported import DiffRendering.GradientIOWrapper;
import Utils.Math.MathHelpers;

struct PBRTDiffuseBSDF : IBSDF, IDifferentiable
{
    float3 albedo;

    [Differentiable]
    float3 eval<S : ISampleGenerator>(const float3 wiLocal, const float3 woLocal, inout S sg)
    {
        return albedo * max(woLocal.z, 0.0f) * M_1_PI;
    }

    bool sample<S : ISampleGenerator>(float3 wiLocal, out float3 wo, out float pdf, out float3 weight, out uint lobeType, inout S sg)
    {
        wo = sample_cosine_hemisphere_concentric(sampleNext2D(sg), pdf);
        weight = albedo;
        lobeType = (uint)LobeType::DiffuseReflection;

        return true;
    }

    float evalPdf(const float3 wiLocal, const float3 woLocal)
    {
        return max(woLocal.z, 0.0f) * M_1_PI;
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

struct PBRTDiffuseMaterialInstance : MaterialInstanceBase, IMaterialInstance
{
    ShadingFrame sf;            ///< Shading frame in world space.
    PBRTDiffuseBSDF bsdf;

    float3 eval<S : ISampleGenerator>(const ShadingData sd, const float3 wo, inout S sg)
    {
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = sf.toLocal(wo);

        if (!isValidHemisphereReflection(sd, sf, wiLocal, woLocal, wo)) return float3(0.f);

        return bsdf.eval(wiLocal, woLocal, sg);
    }

    [Differentiable]
    float3 evalAD<S : ISampleGenerator>(const DiffMaterialData diffData, const ShadingData sd, const float3 wo, inout S sg)
    {
        float3 wiLocal = diffData.sf.toLocal(sd.V);
        float3 woLocal = diffData.sf.toLocal(wo);

        if (!isValidHemisphereReflection(sd, sf, wiLocal, woLocal, wo)) return float3(0.f);

        uint offset = 0;
        float3 albedo = diffData.read<3>(offset);
        PBRTDiffuseBSDF bsdfAD = { albedo };

        return bsdfAD.eval(wiLocal, woLocal, sg);
    }

    bool sample<S : ISampleGenerator>(const ShadingData sd, inout S sg, out BSDFSample result, bool useImportanceSampling = true)
    {
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = {};

        bool valid = bsdf.sample(wiLocal, woLocal, result.pdf, result.weight, result.lobeType, sg);
        result.wo = sf.fromLocal(woLocal);

        if (!isValidHemisphereReflection(sd, sf, wiLocal, woLocal, result.wo) || result.pdf == 0.f) return false;

        return valid;

    }

    float evalPdf(const ShadingData sd, const float3 wo, bool useImportanceSampling = true)
    {
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = sf.toLocal(wo);

        if (!isValidHemisphereReflection(sd, sf, wiLocal, woLocal, wo)) return 0.f;

        return bsdf.evalPdf(wiLocal, woLocal);
    }

    BSDFProperties getProperties(const ShadingData sd)
    {
        BSDFProperties p = {};

        p.guideNormal = sf.N;
        p.roughness = 1.0f;
        p.diffuseReflectionAlbedo = bsdf.albedo;

        return p;
    }

    uint getLobeTypes(const ShadingData sd)
    {
        return (uint)LobeType::DiffuseReflection;
    }
};
