/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
__exported import Rendering.Materials.IBSDF;

/**
 * Unrefracted dirac transmission
 *  The transmission color can be used to tint
 */
struct SimpleBTDF : IBSDF
{
    float3 transmittance; ///< Transmittance colour (i.e. tint)

    __init(float3 transmittance) { this.transmittance = transmittance; }

    float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg) { return float3(0.f); }

    bool sample<S : ISampleGenerator>(const float3 wi, out float3 wo, out float pdf, out float3 weight, out uint lobeType, inout S sg)
    {
        // Default initialization to avoid divergence at returns.
        wo = {};
        weight = {};
        pdf = {};
        lobeType = {};

        if (wi.z > kMinCosTheta)
            return false;

        wo = float3(-wi.x, -wi.y, -wi.z);
        pdf = 0.f;
        weight = transmittance;
        lobeType = (uint)LobeType::DeltaTransmission;
        return true;
    }

    float evalPdf(const float3 wi, const float3 wo) { return 0.f; }

    AlbedoContributions evalAlbedo(const float3 wi, const LobeType lobetype)
    {
        return AlbedoContributions(0.0f, 0.0f, transmittance, 1.0f - transmittance);
    }

    RoughnessInformation getRoughnessInformation(const float3 wi)
    {
        RoughnessInformation r;
        r.roughnessBSDFNotation = float2(0.f, 0.f);
        return r;
    }
};
