/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
__exported import Rendering.Materials.IMaterial;
__exported import Rendering.Materials.HairMaterialInstance;
import Scene.Material.BasicMaterialData;

/** Implementation of Falcor's standard hair/fur material.
 */
struct HairMaterial : MaterialBase, IMaterial
{
    BasicMaterialData data;

    // Specify that the associated `MaterialInstance` type is `HairMaterialInstance`.
    typedef HairMaterialInstance MaterialInstance;

    HairMaterialInstance setupMaterialInstance(const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod, const uint hints)
    {
        SamplerState s = ms.getTextureSampler(header.getDefaultTextureSamplerID());

        HairChiang16Data d = {};

        float16_t IoR = header.getIoR();
        d.IoR = IoR;
        d.eta = sd.frontFacing ? (sd.IoR / IoR) : (IoR / sd.IoR);

        // Sample base color.
        const float3 baseColor = ms.sampleTexture(data.texBaseColor, s, sd.uv, data.baseColor, lod).rgb;

        // Sample the specular texture. This encodes additional BSDF parameters.
        const float4 spec = ms.sampleTexture(data.texSpecular, s, sd.uv, data.specular, lod);

        d.baseColor = baseColor;
        d.betaM = spec.x;
        d.betaN = spec.y;
        d.alpha = spec.z;

        // Compute final shading frame.
        // Do not flip the shading normal for backfacing hits. The BSDF handles transmission from both sides.
        ShadingFrame sf = sd.frame;
        if (isAdjustShadingNormalEnabled(hints))
        {
            adjustShadingNormal(sd, sf);
        }

        return HairMaterialInstance(sf, d);
    }

    [Differentiable]
    HairMaterialInstance setupDiffMaterialInstance(out DiffMaterialData diffData, const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod, const uint hints)
    {
        diffData = DiffMaterialData();
        return no_diff setupMaterialInstance(ms, sd, lod, hints);
    }
};
