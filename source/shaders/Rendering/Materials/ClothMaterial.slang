/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
__exported import Rendering.Materials.IMaterial;
__exported import Rendering.Materials.ClothMaterialInstance;
import Scene.Material.BasicMaterialData;
import Scene.Material.ShadingUtils;
import Scene.Scene;

/** Implementation of a simple cloth material.
 */
struct ClothMaterial : MaterialBase, IMaterial
{
    BasicMaterialData data;

    // Specify that the associated `MaterialInstance` type is `ClothMaterialInstance`.
    typedef ClothMaterialInstance MaterialInstance;

    ClothMaterialInstance setupMaterialInstance(const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod, const uint hints)
    {
        SamplerState s = ms.getTextureSampler(header.getDefaultTextureSamplerID());

        ClothBRDF brdf = {};

        // Sample base color.
        const float3 baseColor = ms.sampleTexture(data.texBaseColor, s, sd.uv, data.baseColor, lod).rgb;

        // Sample the specular texture. This encodes additional BSDF parameters.
        const float4 spec = ms.sampleTexture(data.texSpecular, s, sd.uv, data.specular, lod);

        // Calculate the specular reflectance for dielectrics from the IoR, as in the Disney BSDF [Burley 2015].
        // UE4 uses 0.08 multiplied by a default specular value of 0.5, hence F0=0.04 as default. The default IoR=1.5 gives the same result.
        float f = (header.getIoR() - 1.f) / (header.getIoR() + 1.f);
        float F0 = f * f;

        brdf.diffuseColor = baseColor;
        brdf.subsurfaceColor = float3(0.5f);    // TODO: Add material parameter for this.
        brdf.roughness = spec.g;                // TODO: Linear or squared roughness?
        brdf.f0 = float3(F0);                   // TODO: What's appropriate here?

#if defined(SCENE_DIFFUSE_ALBEDO_MULTIPLIER)
        // DEMO21
        if (SCENE_DIFFUSE_ALBEDO_MULTIPLIER != 1.f)
        {
            brdf.diffuseColor = saturate(brdf.diffuseColor * SCENE_DIFFUSE_ALBEDO_MULTIPLIER);
        }
#endif

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

        return ClothMaterialInstance(sf, brdf);
    }

    [Differentiable]
    ClothMaterialInstance setupDiffMaterialInstance(out DiffMaterialData diffData, const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod, const uint hints)
    {
        diffData = DiffMaterialData();
        return no_diff setupMaterialInstance(ms, sd, lod, hints);
    }
};
