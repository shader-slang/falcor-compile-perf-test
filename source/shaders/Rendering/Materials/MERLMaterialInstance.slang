/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"

__exported import Rendering.Materials.IMaterialInstance;
import Rendering.Materials.BSDFs.DiffuseSpecularBRDF;
import Rendering.Materials.MERLCommon;
import Utils.Math.MathHelpers;
import Scene.Scene;

// Enable to use fitted BRDF approximation for sampling.
#define SAMPLE_FITTED_BRDF 1

// Enable to directly visualize the fitted BRDF instead of the MERL BRDF.
// This is for validation purposes only and should be disabled by default.
#define VISUALIZE_FITTED_BRDF 0

/** Implementation of the BSDF for the measured MERL material.
*/
struct MERLMaterialInstance : MaterialInstanceBase, IMaterialInstance
{
    ShadingFrame sf;    ///< Shading frame in world space.
    uint bufferID;      ///< Buffer ID in material system where BRDF data is stored.
    float3 albedo;      ///< Approximate albedo.
    DiffuseSpecularBRDF fittedBrdf;

    __init(const ShadingFrame sf, const uint bufferID, const float3 albedo, const DiffuseSpecularData extraData)
    {
        this.sf = sf;
        this.bufferID = bufferID;
        this.albedo = albedo;

        // Setup BRDF approximation.
        this.fittedBrdf = DiffuseSpecularBRDF(extraData);
    }

    float3 eval<S : ISampleGenerator>(const ShadingData sd, const float3 wo, inout S sg)
    {
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = sf.toLocal(wo);

        if (!isValidHemisphereReflection(sd, sf, wiLocal, woLocal, wo)) return float3(0.f);

#if VISUALIZE_FITTED_BRDF
        return fittedBrdf.eval(wiLocal, woLocal, sg);
#else
        return evalLocal(wiLocal, woLocal);
#endif
    }

    [Differentiable]
    float3 evalAD<S : ISampleGenerator>(const DiffMaterialData diffData, const ShadingData sd, const float3 wo, inout S sg)
    {
        return float3(0.f);
    }

    bool sample<S : ISampleGenerator>(const ShadingData sd, inout S sg, out BSDFSample result, bool useImportanceSampling = true)
    {
        float3 wiLocal = sf.toLocal(sd.V);

#if SAMPLE_FITTED_BRDF
        float3 woLocal = {};
        if (!fittedBrdf.sample(wiLocal, woLocal, result.pdf, result.weight, result.lobeType, sg)) return false;
        result.wo = sf.fromLocal(woLocal);

        if (!isValidHemisphereReflection(sd, sf, wiLocal, woLocal, result.wo) || result.pdf == 0.f) return false;

#if !VISUALIZE_FITTED_BRDF
        result.weight = evalLocal(wiLocal, woLocal) / result.pdf;
#endif

#else // !SAMPLE_FITTED_BRDF
        // Draw cosine-weighted sample over the hemisphere.
        // TODO: Implement better importance sampling.
        float3 woLocal = sample_cosine_hemisphere_concentric(sampleNext2D(sg), result.pdf); // pdf = cos(theta) / pi
        result.wo = sf.fromLocal(woLocal);

        if (!isValidHemisphereReflection(sd, sf, wiLocal, woLocal, result.wo) || result.pdf == 0.f) return false;

#if VISUALIZE_FITTED_BRDF
        result.weight = fittedBrdf.eval(wiLocal, woLocal, sg) / result.pdf;
#else
        result.weight = evalLocal(wiLocal, woLocal) / result.pdf;
#endif
        result.lobeType = (uint)LobeType::DiffuseReflection;
#endif

        return true;
    }

    float evalPdf(const ShadingData sd, const float3 wo, bool useImportanceSampling = true)
    {
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = sf.toLocal(wo);

        if (!isValidHemisphereReflection(sd, sf, wiLocal, woLocal, wo)) return 0.f;

#if SAMPLE_FITTED_BRDF
        return fittedBrdf.evalPdf(wiLocal, woLocal);
#else
        return woLocal.z * M_1_PI; // pdf = cos(theta) / pi
#endif
    }

    BSDFProperties getProperties(const ShadingData sd)
    {
        BSDFProperties p = {};

        p.guideNormal = sf.N;

        // Compute approximation of the perceived roughness. For now assume a medium roughness.
        p.roughness = fittedBrdf.roughness;

        // Return approximation of the albedos.
        // The BSDF reports only a diffuse reflection lobe, so leaving the other albedos as zero.
        // TODO: Return more accurate albedos.
        p.diffuseReflectionAlbedo = albedo;

        return p;
    }

    uint getLobeTypes(const ShadingData sd)
    {
        return (uint)LobeType::DiffuseReflection | (uint)LobeType::SpecularReflection;
    }


    // Internal helpers

    /** Evaluate the MERL BRDF in the local frame.
        The vectors are assumed to be in the upper hemisphere.
        \param[in] wi Incident direction in the local frame.
        \param[in] wo Outgoing direction in the local frame.
        \return f(wi, wo) * wo.z
    */
    float3 evalLocal(const float3 wi, const float3 wo)
    {
        ByteAddressBuffer brdfData = gScene.materials.getBuffer(bufferID);
        return MERLCommon::eval(wi, wo, brdfData);
    }
};
