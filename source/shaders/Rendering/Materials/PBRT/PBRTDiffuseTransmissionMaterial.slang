/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
__exported import Rendering.Materials.IMaterial;
__exported import Rendering.Materials.PBRT.PBRTDiffuseTransmissionMaterialInstance;
import Scene.Material.BasicMaterialData;
import Scene.Material.ShadingUtils;

struct PBRTDiffuseTransmissionMaterial : MaterialBase, IMaterial
{
    BasicMaterialData data;

    typedef PBRTDiffuseTransmissionMaterialInstance MaterialInstance;

    PBRTDiffuseTransmissionMaterialInstance setupMaterialInstance(const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod, const uint hints)
    {
        SamplerState s = ms.getTextureSampler(header.getDefaultTextureSamplerID());

        const float3 baseColor = ms.sampleTexture(data.texBaseColor, s, sd.uv, data.baseColor, lod).rgb;
        const float3 transColor = ms.sampleTexture(data.texTransmission, s, sd.uv, float4(data.transmission, 1.0f), lod).rgb;

        // Compute final shading frame.
        // Do not flip the shading normal for backfacing hits. The BSDF handles transmission from both sides.
        ShadingFrame sf = sd.frame;
        if (isNormalMappingEnabled(hints))
        {
            float3 encodedNormal = ms.sampleTexture(data.texNormalMap, s, sd.uv, float4(0.f), lod).rgb;
            sf = computeShadingFrameFromNormalMap(sd, data.getNormalMapType(), encodedNormal);
        }
        if (isAdjustShadingNormalEnabled(hints))
        {
            adjustShadingNormal(sd, sf);
        }

        PBRTDiffuseTransmissionMaterialInstance mi = {sf, baseColor, transColor};

        return mi;
    }

    [Differentiable]
    PBRTDiffuseTransmissionMaterialInstance setupDiffMaterialInstance(out DiffMaterialData diffData, const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod, const uint hints)
    {
        diffData = DiffMaterialData();
        return no_diff setupMaterialInstance(ms, sd, lod, hints);
    }
};
