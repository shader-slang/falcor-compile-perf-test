/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"
#include "BCSDFConfig.slangh"

__exported import Rendering.Materials.IMaterialInstance;
__exported import Rendering.Materials.HairChiang16;
import Utils.Math.MathHelpers;

/** Implementation of Falcor's standard hair/fur BSDF.

    The implementation uses the scattering function by Chiang et al. 2016,
    "A Practical and Controllable Hair and Fur Model for Production Path Tracing".

    By overriding the 'BCSDF' define it is possible to select a single-sided
    Lambertian diffuse BRDF for debugging purposes (see BCSDFConfig.slangh).
*/
struct HairMaterialInstance : MaterialInstanceBase, IMaterialInstance
{
    ShadingFrame sf;    ///< Shading frame in world space.
    HairChiang16Data data;

    __init(const ShadingFrame sf, const HairChiang16Data data)
    {
        this.sf = sf;
        this.data = data;
    }

    float3 eval<S : ISampleGenerator>(const ShadingData sd, const float3 wo, inout S sg)
    {
#if BCSDF == HairChiang
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = sf.toLocal(wo);

        HairChiang16 bcsdf = HairChiang16(sd, data);

        return bcsdf.eval(wiLocal, woLocal, sg);
#else
        // Diffuse BRDF
        return data.baseColor * (float)M_1_PI * saturate(dot(sf.N, wo));
#endif
    }

    [Differentiable]
    float3 evalAD<S : ISampleGenerator>(const DiffMaterialData diffData, const ShadingData sd, const float3 wo, inout S sg)
    {
        return float3(0.f);
    }

    bool sample<S : ISampleGenerator>(const ShadingData sd, inout S sg, out BSDFSample result, bool useImportanceSampling = true)
    {
#if BCSDF == HairChiang
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = {};

        HairChiang16 bcsdf = HairChiang16(sd, data);

        bool valid = bcsdf.sample(wiLocal, woLocal, result.pdf, result.weight, result.lobeType, sg);
        result.wo = sf.fromLocal(woLocal);

        return valid;
#else
        // Diffuse BRDF
        float3 woLocal = sample_cosine_hemisphere_concentric(sampleNext2D(sg), result.pdf);
        result.wo = sf.fromLocal(woLocal);
        result.weight = data.baseColor * (float)M_1_PI;
        result.lobeType = (uint)LobeType::DiffuseReflection;
        return true;
#endif
    }

    float evalPdf(const ShadingData sd, const float3 wo, bool useImportanceSampling = true)
    {
#if BCSDF == HairChiang
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = sf.toLocal(wo);

        HairChiang16 bcsdf = HairChiang16(sd, data);

        return bcsdf.evalPdf(wiLocal, woLocal);
#else
        // Diffuse BRDF
        return saturate(dot(sf.N, wo));
#endif
    }

    BSDFProperties getProperties(const ShadingData sd)
    {
        BSDFProperties p = {};

        p.guideNormal = sf.N;

        // Compute approximation of the perceived roughness.
        // It's an open problem how to best do this. For now assume a medium roughness.
        p.roughness = 0.5f;

        // Compute approximation of the albedos.
        // We don't currently have a good way to do this, so divide the energy equally for now.
        // The BSDF reports only diffuse reflection and transmission lobes, so leaving specular albedos as zero.
        // Note the case of black base color is assigned a zero albedo, but presumably there is still some scattering.
        p.diffuseReflectionAlbedo = data.baseColor * 0.5f;
        p.diffuseTransmissionAlbedo = p.diffuseReflectionAlbedo;

        p.flags |= (uint)BSDFProperties::Flags::IsTransmissive;

        return p;
    }

    uint getLobeTypes(const ShadingData sd)
    {
        // The BSDF currently assumes all diffuse lobes (for animal fur).
        // TODO: Update this when lobes are classified as diffuse or specular according to roughness.
        return (uint)LobeType::DiffuseReflection | (uint)LobeType::DiffuseTransmission;
    }
};
