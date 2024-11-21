/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
__exported import Rendering.Materials.IMaterial;
__exported import Rendering.Materials.PBRT.PBRTDielectricMaterialInstance;
import Scene.Material.BasicMaterialData;
import Scene.Material.ShadingUtils;

struct PBRTDielectricMaterial : MaterialBase, IMaterial
{
    BasicMaterialData data;

    typedef PBRTDielectricMaterialInstance MaterialInstance;

    PBRTDielectricMaterialInstance setupMaterialInstance(const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod, const uint hints)
    {
        SamplerState s = ms.getTextureSampler(header.getDefaultTextureSamplerID());

        const float2 roughness = ms.sampleTexture(data.texSpecular, s, sd.uv, data.specular, lod).rg;
        float16_t IoR = header.getIoR();
        float eta = !sd.frontFacing ? 1.0f / IoR : IoR;

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

        PBRTDielectricMaterialInstance mi = {sf, {AnisotropicGGX(roughness), eta}};

        return mi;
    }

    [Differentiable]
    PBRTDielectricMaterialInstance setupDiffMaterialInstance(out DiffMaterialData diffData, const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod, const uint hints)
    {
        diffData = DiffMaterialData();
        return no_diff setupMaterialInstance(ms, sd, lod, hints);
    }
};
