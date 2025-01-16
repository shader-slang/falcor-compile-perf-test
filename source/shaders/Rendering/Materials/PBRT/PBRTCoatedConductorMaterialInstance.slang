/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
__exported import Rendering.Materials.IMaterialInstance;
__exported import Rendering.Materials.AnisotropicGGX;
import Rendering.Materials.LayeredBSDF;
import Rendering.Materials.PBRT.PBRTConductorMaterialInstance;
import Rendering.Materials.PBRT.PBRTDielectricMaterialInstance;

struct PBRTCoatedConductorMaterialInstance : MaterialInstanceBase, IMaterialInstance
{
    ShadingFrame sf;            ///< Shading frame in world space.
    AnisotropicGGX interfaceD;
    AnisotropicGGX conductorD;
    float interfaceEta;
    float3 conductorEta;
    float3 conductorK;
    int maxDepth;

    float3 eval<S : ISampleGenerator>(const ShadingData sd, const float3 wo, inout S sg)
    {
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = sf.toLocal(wo);

        if (!isValidHemisphereReflection(sd, sf, wiLocal, woLocal, wo)) return float3(0.f);

        PBRTDielectricBSDF top = {interfaceD, interfaceEta};
        PBRTConductorBSDF bottom = {DiffMaterialData()};

        return LayeredBSDF(top, bottom).eval(wiLocal, woLocal, sg);
    }

    [Differentiable]
    float3 evalAD<S : ISampleGenerator>(const DiffMaterialData diffData, const ShadingData sd, const float3 wo, inout S sg)
    {
        return float3(0.f);
    }

    bool sample<S : ISampleGenerator>(const ShadingData sd, inout S sg, out BSDFSample result, bool useImportanceSampling = true)
    {
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = {};

        PBRTDielectricBSDF top = {interfaceD, interfaceEta};
        PBRTConductorBSDF bottom = {DiffMaterialData()};

        bool valid = LayeredBSDF(top, bottom).sample(wiLocal, woLocal, result.pdf, result.weight, result.lobeType, sg);
        result.wo = sf.fromLocal(woLocal);

        if (!isValidHemisphereReflection(sd, sf, wiLocal, woLocal, result.wo) || result.pdf == 0.f) return false;

        return valid;
    }

    float evalPdf(const ShadingData sd, const float3 wo, bool useImportanceSampling = true)
    {
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = sf.toLocal(wo);

        if (!isValidHemisphereReflection(sd, sf, wiLocal, woLocal, wo)) return 0.f;

        PBRTDielectricBSDF top = {interfaceD, interfaceEta};
        PBRTConductorBSDF bottom = {DiffMaterialData()};

        return LayeredBSDF(top, bottom).evalPdf(wiLocal, woLocal);
    }

    BSDFProperties getProperties(const ShadingData sd)
    {
        BSDFProperties p = {};

        p.guideNormal = sf.N;
        p.roughness = interfaceD.getIsotropicRoughness();
        p.specularReflectionAlbedo = float3(1.0f);

        return p;
    }

    uint getLobeTypes(const ShadingData sd)
    {
        return (uint)(interfaceD.isSingular() ? LobeType::DeltaReflection : LobeType::SpecularReflection)
             | (uint)(conductorD.isSingular() ? LobeType::DeltaReflection : LobeType::SpecularReflection);
    }
};
