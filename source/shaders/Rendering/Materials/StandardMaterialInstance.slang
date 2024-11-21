/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"

__exported import Rendering.Materials.IMaterialInstance;
__exported import Rendering.Materials.BSDFs.StandardBSDF;
__exported import DiffRendering.SceneGradients;
__exported import DiffRendering.GradientIOWrapper;
import Utils.Math.MathHelpers;
import Rendering.Volumes.PhaseFunction;
import Scene.Material.VolumeProperties;

/** Implementation of Falcor's standard surface BSDF.

    The BSDF has the following lobes:
    - Delta reflection (ideal specular reflection).
    - Specular reflection using a GGX microfacet model.
    - Diffuse reflection using Disney's diffuse BRDF.
    - Delta transmission (ideal specular transmission).
    - Specular transmission using a GGX microfacet model.
    - Diffuse transmission.

    The BSDF is a linear combination of the above lobes.
*/
struct StandardMaterialInstance : MaterialInstanceBase, IMaterialInstance
{
    ShadingFrame sf;            ///< Shading frame in world space.
    StandardBSDFData data;      ///< BSDF parameters.
    float3 emission;            ///< Radiance emitted in the incident direction (wi).

    __init(const ShadingFrame sf, const StandardBSDFData data, const float3 emission)
    {
        this.sf = sf;
        this.data = data;
        this.emission = emission;
    }

    float3 eval<S : ISampleGenerator>(const ShadingData sd, const float3 wo, inout S sg)
    {
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = sf.toLocal(wo);

        if (!isValidHemisphereReflectionOrTransmission(sd, sf, wiLocal, woLocal, wo)) return float3(0.f);

        StandardBSDF bsdf = StandardBSDF(wiLocal, sd.mtl, data);

        return bsdf.eval(wiLocal, woLocal, sg);
    }

    [Differentiable]
    float3 evalAD<S : ISampleGenerator>(const DiffMaterialData diffData, const ShadingData sd, const float3 wo, inout S sg)
    {
        float3 wiLocal = diffData.sf.toLocal(sd.V);
        float3 woLocal = diffData.sf.toLocal(wo);

        if (!isValidHemisphereReflectionOrTransmission(sd, sf, wiLocal, woLocal, wo)) return float3(0.f);

        StandardBSDFData dataAD = StandardBSDFData(diffData);
        StandardBSDF bsdfAD = StandardBSDF(wiLocal, sd.mtl, dataAD);

        return bsdfAD.eval(wiLocal, woLocal, sg);
    }

    bool sample<S : ISampleGenerator>(const ShadingData sd, inout S sg, out BSDFSample result, bool useImportanceSampling = true)
    {
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = {};

        if (!useImportanceSampling)
        {
            if (!sampleReference(sd, wiLocal, woLocal, result.pdf, result.weight, result.lobeType, sg))
                return false;
        }
        else
        {
            StandardBSDF bsdf = StandardBSDF(wiLocal, sd.mtl, data);
            if (!bsdf.sample(wiLocal, woLocal, result.pdf, result.weight, result.lobeType, sg))
                return false;
        }
        result.wo = sf.fromLocal(woLocal);

        if (!isValidHemisphereReflectionOrTransmission(sd, sf, wiLocal, woLocal, result.wo)) return false;

        return true;
    }

    float evalPdf(const ShadingData sd, const float3 wo, bool useImportanceSampling = true)
    {
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = sf.toLocal(wo);

        if (!isValidHemisphereReflectionOrTransmission(sd, sf, wiLocal, woLocal, wo)) return 0.f;

        if (!useImportanceSampling)
        {
            return evalPdfReference(sd, wiLocal, woLocal);
        }
        else
        {
            StandardBSDF bsdf = StandardBSDF(wiLocal, sd.mtl, data);
            return bsdf.evalPdf(wiLocal, woLocal);
        }
    }

    BSDFProperties getProperties(const ShadingData sd)
    {
        BSDFProperties p = {};

        p.guideNormal = sf.N;
        p.emission = emission;

        // Clamp roughness so it's representable of what is actually used in StandardBSDF.
        // Roughness^2 below kMinGGXAlpha is used to indicate perfectly smooth surfaces.
        float alpha = data.roughness * data.roughness;
        p.roughness = alpha < kMinGGXAlpha ? 0.f : data.roughness;

        // Compute approximation of the albedos.
        // For now use the blend weights and colors, but this should be improved to better numerically approximate the integrals.
        p.diffuseReflectionAlbedo = (1.f - data.diffuseTransmission) * (1.f - data.specularTransmission) * data.diffuse;
        p.diffuseTransmissionAlbedo = data.diffuseTransmission * (1.f - data.specularTransmission) * data.transmission;
        p.specularReflectionAlbedo = (1.f - data.specularTransmission) * data.specular;
        p.specularTransmissionAlbedo = data.specularTransmission * data.specular;

        // Pass on our specular reflectance field unmodified.
        p.specularReflectance = data.specular;

        if (data.diffuseTransmission > 0.f || data.specularTransmission > 0.f) p.flags |= (uint)BSDFProperties::Flags::IsTransmissive;

        return p;
    }

    uint getLobeTypes(const ShadingData sd)
    {
        return StandardBSDF::getLobeTypes(data);
    }

    bool hasVolumeProperties()
    {
        return data.hasEntryPointVolumeProperties;
    }

    VolumeProperties getVolumeProperties()
    {
        VolumeProperties result;
        if (data.hasSigmaSGreaterZero)
        {
            // Convert color to sigmaA. See MxLayeredMaterialInstance fro details
            const float g = data.volumeAnsiotropy;
            const float3 c = data.diffuse;
            const float3 s = 4.09712f + 4.20863f*c - sqrt(9.59217f + 41.6808f * c + 17.7126 * c * c);
            const float3 alpha = (1.f - s*s) / (1.f - g * s*s);

            result.sigmaA = select(alpha > 0.f, (1.f - alpha) * (data.volumeScattering) / alpha, 0.f);
        }
        else
            result.sigmaA = 0.f;

        result.sigmaS = data.volumeScattering;
        result.phaseFunction = NullPhaseFunction(); // g is taken from the volume and set in InternalPathTracer
        return result;

    }

    // Additional functions

    /** Reference implementation that uses cosine-weighted hemisphere sampling.
        This is for testing purposes only.
        \param[in] sd Shading data.
        \param[in] wi Incident direction in local space.
        \param[out] wo Outgoing direction in local space.
        \param[out] pdf pdf with respect to solid angle for sampling outgoing direction wo (0 if a delta event is sampled).
        \param[out] weight Sample weight f(wi, wo) * dot(wo, n) / pdf(wo).
        \param[out] lobeType Sampled lobeType (see LobeType).
        \param[in,out] sg Sample generator.
        \return True if a sample was generated, false otherwise.
    */
    bool sampleReference<S : ISampleGenerator>(const ShadingData sd, const float3 wi, out float3 wo, out float pdf, out float3 weight, out uint lobeType, inout S sg)
    {
        weight = {};
        lobeType = {};

        const bool isTransmissive = (getLobeTypes(sd) & (uint)LobeType::Transmission) != 0;

        wo = sample_cosine_hemisphere_concentric(sampleNext2D(sg), pdf); // pdf = cos(theta) / pi

        if (isTransmissive)
        {
            if (sampleNext1D(sg) < 0.5f)
            {
                wo.z = -wo.z;
            }
            pdf *= 0.5f;
            if (min(abs(wi.z), abs(wo.z)) < kMinCosTheta || pdf == 0.f) return false;
        }
        else
        {
            if (min(wi.z, wo.z) < kMinCosTheta || pdf == 0.f) return false;
        }

        StandardBSDF bsdf = StandardBSDF(wi, sd.mtl, data);

        weight = bsdf.eval(wi, wo, sg) / pdf;
        lobeType = (uint)(wo.z > 0.f ? LobeType::DiffuseReflection : LobeType::DiffuseTransmission);

        return true;
    }

    /** Evaluates the directional pdf for sampling the given direction using the reference implementation.
        \param[in] sd Shading data.
        \param[in] wi Incident direction in local space.
        \param[in] wo Outgoing direction in local space.
        \return PDF with respect to solid angle for sampling direction wo.
    */
    float evalPdfReference(const ShadingData sd, const float3 wi, const float3 wo)
    {
        const bool isTransmissive = (getLobeTypes(sd) & (uint)LobeType::Transmission) != 0;

        if (isTransmissive)
        {
            if (min(abs(wi.z), abs(wo.z)) < kMinCosTheta) return 0.f;
            return 0.5f * wo.z * M_1_PI; // pdf = 0.5 * cos(theta) / pi
        }
        else
        {
            if (min(wi.z, wo.z) < kMinCosTheta) return 0.f;
            return wo.z * M_1_PI; // pdf = cos(theta) / pi
        }
    }
};
