/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
__exported import Rendering.Materials.IMaterial;
__exported import Rendering.Materials.MERLMixMaterialInstance;
import Scene.Material.MERLMixMaterialData;
import Scene.Scene;

/** Implementation of measured MERLMix material.
 */
struct MERLMixMaterial : MaterialBase, IMaterial
{
    MERLMixMaterialData data;

    // Specify that the associated `MaterialInstance` type is `MERLMixMaterialInstance`.
    typedef MERLMixMaterialInstance MaterialInstance;

    MERLMixMaterialInstance setupMaterialInstance(const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod, const uint hints)
    {
        let explicitLod = ExplicitLodTextureSampler(0.f);

        // Compute final shading frame.
        ShadingFrame sf = sd.frame;
        if (isNormalMappingEnabled(hints))
        {
            SamplerState s = ms.getTextureSampler(header.getDefaultTextureSamplerID());
            float3 encodedNormal = ms.sampleTexture(data.texNormalMap, s, sd.uv, float4(0.f), lod).rgb;
            sf = computeShadingFrameFromNormalMap(sd, data.getNormalMapType(), encodedNormal);
        }
        flipShadingNormal(sd, sf);
        if (isAdjustShadingNormalEnabled(hints))
        {
            adjustShadingNormal(sd, sf);
        }

        // Fetch the BRDF index to use at the shading point.
        // The index map is assumed to be in 8-bit unorm format and we use the red channel scaled by 255 as index.
        // The index wraps around modulo the actual BRDF count.
        uint brdfIndex = 0;
        {
            SamplerState s = ms.getTextureSampler(data.getIndexSamplerID());
            float indexUnorm = ms.sampleTexture(data.texIndexMap, s, sd.uv, float4(0.f), explicitLod).r;
            brdfIndex = (uint)(indexUnorm * 255.f) % data.brdfCount;
        }
        uint byteOffset = brdfIndex * data.byteStride;

        // Evaluate the albedo for the current incident (view) direction.
        float3 albedo = {};
        float cosTheta = dot(sf.N, sd.V);

        if (cosTheta >= kMinCosTheta)
        {
            // The albedo is precomputed in a lookup table, which we sample using piecewise linear interpolation.
            // The columns store precomputed albedo sampled at cosTheta = (1,2,...N)/N, where N is the LUT size.
            // We offset the texture coordinate by -0.5/N to get correct interpolation of the two nearest samples.
            // The rows store one lookup table per BRDF index. We sample the midpoint of the texels vertically.
            float u = cosTheta - 0.5f / MERLMixMaterialData::kAlbedoLUTSize;
            float v = (brdfIndex + 0.5f) / (float)data.brdfCount;

            SamplerState s = ms.getTextureSampler(data.getLUTSamplerID());
            albedo = ms.sampleTexture(data.texAlbedoLUT, s, float2(u, v), float4(0.5f), explicitLod).rgb;
        }

        // Load extra data for sampling.
        ByteAddressBuffer brdfData = gScene.materials.getBuffer(data.bufferID);
        uint extraDataByteOffset = data.extraDataOffset + brdfIndex * data.extraDataStride;
        DiffuseSpecularData extraData = brdfData.Load<DiffuseSpecularData>(extraDataByteOffset);

        return MERLMixMaterialInstance(sf, data.bufferID, byteOffset, albedo, brdfIndex, extraData);
    }

    [Differentiable]
    MERLMixMaterialInstance setupDiffMaterialInstance(out DiffMaterialData diffData, const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod, const uint hints)
    {
        diffData = DiffMaterialData();
        return no_diff setupMaterialInstance(ms, sd, lod, hints);
    }
};
