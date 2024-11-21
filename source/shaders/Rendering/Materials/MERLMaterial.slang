/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
__exported import Rendering.Materials.IMaterial;
__exported import Rendering.Materials.MERLMaterialInstance;
import Scene.Material.MERLMaterialData;

/** Implementation of measured MERL material.
 */
struct MERLMaterial : MaterialBase, IMaterial
{
    MERLMaterialData data;

    // Specify that the associated `MaterialInstance` type is `MERLMaterialInstance`.
    typedef MERLMaterialInstance MaterialInstance;

    MERLMaterialInstance setupMaterialInstance(const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod, const uint hints)
    {
        // Compute final shading frame.
        ShadingFrame sf = sd.frame;
        flipShadingNormal(sd, sf);
        if (isAdjustShadingNormalEnabled(hints))
        {
            adjustShadingNormal(sd, sf);
        }

        // Evaluate the albedo for the current incident (view) direction.
        float3 albedo = {};
        float cosTheta = dot(sf.N, sd.V);

        if (cosTheta >= kMinCosTheta)
        {
            // The albedo is precomputed in a lookup table, which we sample using piecewise linear interpolation.
            // The precompued albedo values are sampled at cosTheta = (1,2,...N)/N, where N is the LUT size.
            // We offset the texture coordinate by -0.5/N to get correct interpolation of the two nearest samples.
            float u = cosTheta - 0.5f / MERLMaterialData::kAlbedoLUTSize;

            SamplerState s = ms.getTextureSampler(data.samplerID);
            let explicitLod = ExplicitLodTextureSampler(0.f);
            albedo = ms.sampleTexture(data.texAlbedoLUT, s, float2(u, 0.5f), float4(0.5f), explicitLod).rgb;
        }

        return MERLMaterialInstance(sf, data.bufferID, albedo, data.extraData);
    }

    [Differentiable]
    MERLMaterialInstance setupDiffMaterialInstance(out DiffMaterialData diffData, const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod, const uint hints)
    {
        diffData = DiffMaterialData();
        return no_diff setupMaterialInstance(ms, sd, lod, hints);
    }
};
