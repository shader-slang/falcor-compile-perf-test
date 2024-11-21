/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
__exported import Rendering.Materials.IMaterial;
__exported import Rendering.Materials.PBRT.PBRTDiffuseMaterialInstance;
import Scene.Material.PBRT.PBRTDiffuseMaterialParamLayout;
import Scene.Material.BasicMaterialData;
import Scene.Material.ShadingUtils;

struct PBRTDiffuseMaterial : MaterialBase, IMaterial
{
    BasicMaterialData data;

    typedef PBRTDiffuseMaterialInstance MaterialInstance;

    PBRTDiffuseMaterialInstance setupMaterialInstance(const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod, const uint hints)
    {
        SamplerState s = ms.getTextureSampler(header.getDefaultTextureSamplerID());

        const float3 baseColor = ms.sampleTexture(data.texBaseColor, s, sd.uv, data.baseColor, lod).rgb;

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

        PBRTDiffuseMaterialInstance mi = {sf, {baseColor}};

        return mi;
    }

    [Differentiable]
    PBRTDiffuseMaterialInstance setupDiffMaterialInstance(out DiffMaterialData diffData, const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod, const uint hints)
    {
        SamplerState s = ms.getTextureSampler(header.getDefaultTextureSamplerID());

        uint hashIndex = hashFunction(sd.threadID, gSceneGradients.getHashSize(GradientType::Material));
        GradientIOWrapper gradIO = GradientIOWrapper(GradientType::Material, sd.materialGradOffset, hashIndex);

        // TODO: Handle texture gradients.
        const float4 _baseColor = no_diff ms.sampleTexture(data.texBaseColor, s, sd.uv, data.baseColor, lod);
        float3 baseColor = gradIO.getFloat(_baseColor.rgb, PBRTDiffuseMaterialParamLayout.baseColor);

        // Compute final shading frame.
        ShadingFrame sf = sd.frame;
        if (isNormalMappingEnabled(hints))
        {
            float3 encodedNormal = ms.sampleTexture(data.texNormalMap, s, sd.uv, float4(0.f), lod).rgb;
            // TODO: Allow differentiation here.
            // We don't support it because differentiating normal maps is harder due to additional discontinuities.
            sf = no_diff computeShadingFrameFromNormalMap(sd, data.getNormalMapType(), encodedNormal);
        }
        flipShadingNormal(sd, sf);
        if (isAdjustShadingNormalEnabled(hints))
        {
            no_diff adjustShadingNormal(sd, sf);
        }

        PBRTDiffuseMaterialInstance mi = { sf, {baseColor} };

        diffData.sf = sf;
        uint offset = 0;
        diffData.write(baseColor, offset);

        return mi;
    }
};
