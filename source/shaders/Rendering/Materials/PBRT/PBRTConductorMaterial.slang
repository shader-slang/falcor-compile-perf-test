/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
__exported import Rendering.Materials.IMaterial;
__exported import Rendering.Materials.PBRT.PBRTConductorMaterialInstance;
import Scene.Material.PBRT.PBRTConductorMaterialParamLayout;
import Scene.Material.BasicMaterialData;
import Scene.Material.ShadingUtils;

struct PBRTConductorMaterial : MaterialBase, IMaterial
{
    BasicMaterialData data;

    typedef PBRTConductorMaterialInstance MaterialInstance;

    PBRTConductorMaterialInstance setupMaterialInstance(const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod, const uint hints)
    {
        SamplerState s = ms.getTextureSampler(header.getDefaultTextureSamplerID());

        // Abuse these colors and assume eta = baseColor, k = transColor
        const float3 baseColor = ms.sampleTexture(data.texBaseColor, s, sd.uv, data.baseColor, lod).rgb;
        const float3 transColor = ms.sampleTexture(data.texTransmission, s, sd.uv, float4(data.transmission, 1.0f), lod).rgb;
        const float2 roughness = ms.sampleTexture(data.texSpecular, s, sd.uv, data.specular, lod).rg;

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

        PBRTConductorMaterialInstance mi = {sf, {AnisotropicGGX(roughness), baseColor, transColor}};

        return mi;
    }

    [Differentiable]
    PBRTConductorMaterialInstance setupDiffMaterialInstance(out DiffMaterialData diffData, const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod, const uint hints)
    {
        SamplerState s = ms.getTextureSampler(header.getDefaultTextureSamplerID());

        uint hashIndex = hashFunction(sd.threadID, gSceneGradients.getHashSize(GradientType::Material));
        GradientIOWrapper gradIO = GradientIOWrapper(GradientType::Material, sd.materialGradOffset, hashIndex);

        // Abuse these colors and assume eta = baseColor, k = transColor
        // TODO: Handle texture gradients.
        const float4 _baseColor = no_diff ms.sampleTexture(data.texBaseColor, s, sd.uv, data.baseColor, lod);
        float3 baseColor = gradIO.getFloat(_baseColor.rgb, PBRTConductorMaterialParamLayout.baseColor);
        const float4 _transColor = no_diff ms.sampleTexture(data.texTransmission, s, sd.uv, float4(data.transmission, 1.0f), lod);
        float3 transColor = gradIO.getFloat(_transColor.rgb, PBRTConductorMaterialParamLayout.transmissionColor);
        const float4 _roughness = no_diff ms.sampleTexture(data.texSpecular, s, sd.uv, data.specular, lod);
        float2 roughness = gradIO.getFloat(_roughness.rg, PBRTConductorMaterialParamLayout.roughness);

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

        PBRTConductorMaterialInstance mi = { sf, { AnisotropicGGX(roughness), baseColor, transColor } };
        diffData.sf = sf;

        uint offset = 0;
        // 0--3: eta (baseColor)
        diffData.write(baseColor, offset);
        // 3--6: k (transColor)
        diffData.write(transColor, offset);
        // 6--8: roughness
        diffData.write(roughness, offset);

        return mi;
    }
};
