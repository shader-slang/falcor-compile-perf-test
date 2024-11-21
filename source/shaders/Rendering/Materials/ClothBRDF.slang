/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"

import Scene.ShadingData;
import Utils.Math.MathHelpers;
__exported import Rendering.Materials.IBSDF;

// Uncomment to enable subsurface scattering approximation.
// Note we need framework support for supplying a subsurface color before it's useful.
//#define MATERIAL_HAS_SUBSURFACE_COLOR

/** Cloth BRDF based on:
    - Neubelt and Pettineo 2013, "Crafting a Next-gen Material Pipeline for The Order: 1886".
    - Estevez and Kulla 2017, "Production Friendly Microfacet Sheen BRDF".
    - Google Filament.
*/
struct ClothBRDF : IBSDF
{
    float roughness;            ///< Linear roughness.
    float3 f0;                  ///< Specular reflectance at normal incidence.
    float3 diffuseColor;        ///< Diffuse albedo.
    float3 subsurfaceColor;     ///< Subsurface color.

    // Implementation of IBSDF interface

    float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg)
    {
        if (min(wi.z, wo.z) < kMinCosTheta) return float3(0.f);

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
        if (min(wi.z, wo.z) < kMinCosTheta) return 0.f;

        return M_1_PI * wo.z;
    }

    AlbedoContributions evalAlbedo(const float3 wi, const LobeType lobetype)
    {
        // this is not entirely correct, we need to include the sheen as well
        return AlbedoContributions(diffuseColor, 1.f - diffuseColor, 0.f, 0.f);
    }

    RoughnessInformation getRoughnessInformation(const float3 wi)
    {
        RoughnessInformation r;
        r.roughnessBSDFNotation = (roughness + 1.f) * 0.5f;
        return r;
    }

    // Additional functions

    /** Returns f(wi, wo) * cos(theta_o) / p(wi, wo) = f(wi, wo) * pi.
        Both incident and outgoing direction are assumed to be in the positive hemisphere.
    */
    float3 evalWeight(float3 wi, float3 wo)
    {
        float3 h = normalize(wi + wo);
        float NoO = wo.z;
        float NoI = wi.z;
        float NoH = h.z;

        // Specular BRDF
        float D = D_Sheen(roughness, NoH);
        float V = V_Neubelt(NoI, NoO);
        float3 F = f0;
        float3 Fr = (D * V) * F;

        // Diffuse BRDF
        float diffuse = diffuseLambert();
#if defined(MATERIAL_HAS_SUBSURFACE_COLOR)
        // Energy conservative wrapped diffuse to simulate subsurface scattering.
        diffuse *= diffuseWrapped(NoO, 0.5);
#endif

        // Note: Currently not multiplying the diffuse term by the Fresnel term as discussed in
        // Neubelt and Pettineo 2013, "Crafting a Next-gen Material Pipeline for The Order: 1886".
        float3 Fd = diffuse * diffuseColor;

#if defined(MATERIAL_HAS_SUBSURFACE_COLOR)
        // Cheap subsurface scattering approximation.
        Fd *= saturate(subsurfaceColor + NoO);
        float3 weight = Fd / NoO + Fr; // Note: Remove NoO from diffuse again as it'll be applied later.
#else
        float3 weight = (Fd + Fr);
#endif

        return weight * M_PI;
    }

    /** Normal distribution function for cloth.
        Based on Estevez and Kulla 2017, "Production Friendly Microfacet Sheen BRDF".
        \param[in] r Roughness.
        \param[in] NoH Dot product between shading normal and half vector.
        \return Evaluated NDF.
    */
    float D_Sheen(float r, float NoH)
    {
        float invAlpha = 1.f / r;
        float cos2h = NoH * NoH;
        float sin2h = max(1.f - cos2h, 0.0078125f);
        return (2.f + invAlpha) * pow(sin2h, invAlpha * 0.5f) / (2.f * M_PI);
    }

    /** Microfacet visibility function for cloth.
        Based on Neubelt and Pettineo 2013, "Crafting a Next-gen Material Pipeline for The Order: 1886".
    */
    float V_Neubelt(float NoI, float NoO)
    {
        return 1.f / (4.f * (NoO + NoI - NoO * NoI));
    }

    float diffuseLambert()
    {
        return M_1_PI;
    }

    /** Energy conserving wrapped diffuse term, does *not* include the divide by pi.
    */
    float diffuseWrapped(float NoO, float w)
    {
        return saturate((NoO + w) / ((1.f + w) * (1.f + w)));
    }
};
