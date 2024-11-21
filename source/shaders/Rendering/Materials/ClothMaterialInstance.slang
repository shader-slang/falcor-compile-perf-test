/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"

__exported import Rendering.Materials.IMaterialInstance;
__exported import Rendering.Materials.ClothBRDF;
import Utils.Math.MathHelpers;

/** Implementation of a simple cloth BSDF.
*/
struct ClothMaterialInstance : MaterialInstanceBase, IMaterialInstance
{
    ShadingFrame sf;            ///< Shading frame in world space.
    ClothBRDF brdf;

    __init(const ShadingFrame sf, const ClothBRDF brdf)
    {
        this.sf = sf;
        this.brdf = brdf;
    }

    float3 eval<S : ISampleGenerator>(const ShadingData sd, const float3 wo, inout S sg)
    {
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = sf.toLocal(wo);

        if (!isValidHemisphereReflection(sd, sf, wiLocal, woLocal, wo)) return float3(0.f);

        return brdf.eval(wiLocal, woLocal, sg);
    }

    [Differentiable]
    float3 evalAD<S : ISampleGenerator>(const DiffMaterialData diffData, const ShadingData sd, const float3 wo, inout S sg)
    {
        return float3(0.f);
    }

    bool sample<S : ISampleGenerator>(const ShadingData sd, inout S sg, out BSDFSample result, bool useImportanceSampling = true)
    {
        return sampleReference(sd, sg, result);
    }

    float evalPdf(const ShadingData sd, const float3 wo, bool useImportanceSampling = true)
    {
        return evalPdfReference(sd, wo);
    }

    BSDFProperties getProperties(const ShadingData sd)
    {
        BSDFProperties p = {};

        p.guideNormal = sf.N;

        // Compute approximation of the perceived roughness.
        // It's an open problem how to best do this. For now approximate based on the sheen roughness.
        p.roughness = (brdf.roughness + 1.f) * 0.5f;

        // Compute approximation of the albedos.
        // For now assume all diffuse reflection, this should be improved as there is some directionality.
        p.diffuseReflectionAlbedo = brdf.diffuseColor;

        return p;
    }

    uint getLobeTypes(const ShadingData sd)
    {
        return (uint)LobeType::DiffuseReflection;
    }


    // Additional functions

    /** Reference implementation that uses cosine-weighted hemisphere sampling.
        This is for testing purposes only. Note that transmission is not sampled.
        \param[in] sd Shading data.
        \param[in] sg Sample generator.
        \param[out] result Generated sample. Only valid if true is returned.
        \return True if a sample was generated, false otherwise.
    */
    bool sampleReference<S : ISampleGenerator>(const ShadingData sd, inout S sg, out BSDFSample result)
    {
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = sample_cosine_hemisphere_concentric(sampleNext2D(sg), result.pdf); // pdf = cos(theta) / pi
        result.wo = sf.fromLocal(woLocal);

        if (!isValidHemisphereReflection(sd, sf, wiLocal, woLocal, result.wo) || result.pdf == 0.f) return false;

        result.weight = brdf.eval(wiLocal, woLocal, sg) / result.pdf;
        result.lobeType = (uint)LobeType::DiffuseReflection;

        return true;
    }

    /** Evaluates the directional pdf for sampling the given direction using the reference implementation.
        \param[in] sd Shading data.
        \param[in] wo Outgoing direction.
        \return PDF with respect to solid angle for sampling direction wo (0 for delta events).
    */
    float evalPdfReference(const ShadingData sd, const float3 wo)
    {
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = sf.toLocal(wo);

        if (!isValidHemisphereReflection(sd, sf, wiLocal, woLocal, wo)) return 0.f;

        return woLocal.z * M_1_PI; // pdf = cos(theta) / pi
    }
};
