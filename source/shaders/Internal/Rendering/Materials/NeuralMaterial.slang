/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
#include "Internal/Scene/Materials/NeuralMaterialTypes.slangh"

__exported import Rendering.Materials.IMaterial;
__exported import Internal.Rendering.Materials.NeuralMaterialInstance;
import Internal.Scene.Materials.NeuralMaterialData;
import NeuralMaterialImportanceSampling;

/**
 * Implementation of neural material.
 */
struct NeuralMaterial : MaterialBase, IMaterial
{
    // MaterialHeader header is stored here (see MaterialBase).
    NeuralMaterialData data;

    // Specify that the associated `MaterialInstance` type is `NeuralMaterialInstance`.
    typedef NeuralMaterialInstance MaterialInstance;

    NeuralMaterialInstance setupMaterialInstance(const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod, const uint hints)
    {
        // Compute final shading frame.
        ShadingFrame sf = sd.frame;
        flipShadingNormal(sd, sf);
        if (isAdjustShadingNormalEnabled(hints))
        {
            adjustShadingNormal(sd, sf);
        }

        float4 latent0 = {};
        float4 latent1 = {};

#if NEURAL_MATERIAL_TYPE == NEURAL_MATERIAL_TYPE_LATENT_TEXTURES || NEURAL_MATERIAL_TYPE == NEURAL_MATERIAL_TYPE_WITH_SHFRAME_EXTENDED
        // Sample latent textures (if used).
        SamplerState s = ms.getTextureSampler(data.getLatentSamplerID());
        latent0 = ms.sampleTexture(data.texNeural[0], s, sd.uv, float4(0), lod);
        latent1 = ms.sampleTexture(data.texNeural[1], s, sd.uv, float4(0), lod);
#endif

        // Evaluate the albedo for the current incident (view) direction.
        float3 albedo = {};
        float cosTheta = dot(sf.N, sd.V);

        if (cosTheta >= kMinCosTheta)
        {
            albedo = float3(0.5f); // Approximation
        }

        // Fetch importance sampling parameters from MLP.
        float3 alpha = float3(0.f);
        float2 slopeSpec = float2(0.f);
        float2 slopeDiff = float2(0.f);
        float2 weights = float2(0.f);
#ifdef NEURAL_MATERIAL_USE_PROXY_IMPORTANCE_SAMPLING
        evalProxyImportanceSamplingParameters(
            data.paramsByteOffsetImportanceSampling1, latent0, latent1, sf.toLocal(sd.V), alpha, slopeSpec, slopeDiff, weights
        );
#endif
        return NeuralMaterialInstance(
            sf,
            data.paramsByteOffsetEvaluation,
            data.paramsByteOffsetShFrame,
            data.paramsByteOffsetImportanceSampling1,
            data.paramsByteOffsetImportanceSampling2,
            albedo,
            latent0,
            latent1,
            alpha,
            slopeSpec,
            slopeDiff,
            weights.x
        );
    }

    [Differentiable]
    NeuralMaterialInstance setupDiffMaterialInstance(
        out DiffMaterialData diffData,
        const MaterialSystem ms,
        const ShadingData sd,
        const ITextureSampler lod,
        const uint hints
    )
    {
        diffData = DiffMaterialData();
        return no_diff setupMaterialInstance(ms, sd, lod, hints);
    }
};
