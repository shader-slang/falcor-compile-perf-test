/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
__exported import Rendering.Materials.IMaterial;
__exported import Rendering.Materials.RGLMaterialInstance;
import Rendering.Materials.RGLCommon;
import Scene.Material.RGLMaterialData;
import Scene.Scene;

/** Implementation of measured RGL material.
 */
struct RGLMaterial : MaterialBase, IMaterial
{
    RGLMaterialData data;

    // Specify that the associated `MaterialInstance` type is `RGLMaterialInstance`.
    typedef RGLMaterialInstance MaterialInstance;

    RGLMaterialInstance setupMaterialInstance(const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod, const uint hints)
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
        float2 slice = {};
        float sigmaEval = 0.0f;

        if (cosTheta >= kMinCosTheta)
        {
            // The albedo is precomputed in a lookup table, which we sample using piecewise linear interpolation.
            // The precompued albedo values are sampled at cosTheta = (1,2,...N)/N, where N is the LUT size.
            // We offset the texture coordinate by -0.5/N to get correct interpolation of the two nearest samples.
            float u = cosTheta - 0.5f / RGLMaterialData::kAlbedoLUTSize;

            SamplerState s = ms.getTextureSampler(data.samplerID);
            let explicitLod = ExplicitLodTextureSampler(0.f);
            albedo = ms.sampleTexture(data.texAlbedoLUT, s, float2(u, 0.5f), float4(0.5f), explicitLod).rgb;

            // Precompute the unwarped coordinates from wi to the UV coordinates defined by the measured phi/theta angles.
            let sigma = Brick2D.fromManagedBuffers(data.sigmaSize, data.sigmaBufID);
            let warp = CoordinateWarp.fromManagedBuffers(uint2(data.phiSize, data.thetaSize), data.phiBufID, data.thetaBufID);
            float3 wiLocal = sf.toLocal(sd.V);
            float2 sphericalI = toSpherical(wiLocal);
            float2 unitI = sphericalToUnit(sphericalI);
            slice = warp.warp(sphericalI.yx);
            // Pre-evaluate projected area of microfacets (wi is fixed, and so is sigma).
            sigmaEval = sigma.eval(unitI);
        }

        return RGLMaterialInstance(sf, slice, sigmaEval, albedo, data);
    }

    [Differentiable]
    RGLMaterialInstance setupDiffMaterialInstance(out DiffMaterialData diffData, const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod, const uint hints)
    {
        diffData = DiffMaterialData();
        return no_diff setupMaterialInstance(ms, sd, lod, hints);
    }
};
