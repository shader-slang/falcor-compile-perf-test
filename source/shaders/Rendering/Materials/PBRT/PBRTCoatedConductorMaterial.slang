/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
__exported import Rendering.Materials.IMaterial;
__exported import Rendering.Materials.PBRT.PBRTCoatedConductorMaterialInstance;
import Scene.Material.BasicMaterialData;
import Scene.Material.ShadingUtils;

struct PBRTCoatedConductorMaterial : MaterialBase, IMaterial
{
    BasicMaterialData data;

    typedef PBRTCoatedConductorMaterialInstance MaterialInstance;

    PBRTCoatedConductorMaterialInstance setupMaterialInstance(const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod, const uint hints)
    {
        SamplerState s = ms.getTextureSampler(header.getDefaultTextureSamplerID());

        // Abuse these colors and assume conductor.eta = baseColor, conductor.k = transColor
        const float3 baseColor = ms.sampleTexture(data.texBaseColor, s, sd.uv, data.baseColor, lod).rgb;
        const float3 transColor = ms.sampleTexture(data.texTransmission, s, sd.uv, float4(data.transmission, 1.0f), lod).rgb;
        // RG: Interface X&Y roughness
        // BA: Conductor X&Y roughness
        const float4 roughnesses = ms.sampleTexture(data.texSpecular, s, sd.uv, data.specular, lod).rgba;
        float eta = 1.0f / header.getIoR();

        // Compute final shading frame.
        ShadingFrame sf = sd.frame;
        if (isNormalMappingEnabled(hints))
        {
            float3 encodedNormal = ms.sampleTexture(data.texNormalMap, s, sd.uv, float4(0.f), lod).rgb;
            sf = computeShadingFrameFromNormalMap(sd, data.getNormalMapType(), encodedNormal);
        }
        flipShadingNormal(sd, sf);
        if (isAdjustShadingNormalEnabled(hints))
        {
            adjustShadingNormal(sd, sf);
        }

        PBRTCoatedConductorMaterialInstance mi = {sf, AnisotropicGGX(roughnesses.rg), AnisotropicGGX(roughnesses.ba), eta, baseColor, transColor, 32};

        return mi;
    }

    [Differentiable]
    PBRTCoatedConductorMaterialInstance setupDiffMaterialInstance(out DiffMaterialData diffData, const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod, const uint hints)
    {
        diffData = DiffMaterialData();
        return no_diff setupMaterialInstance(ms, sd, lod, hints);
    }
};
