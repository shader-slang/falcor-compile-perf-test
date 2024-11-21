/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import Rendering.Materials.Fresnel;
import Rendering.Materials.IMaterialInstance;
import Rendering.Materials.IsotropicGGX;
import RenderPasses.Shared.Denoising.NRDConstants;
import Scene.Scene;
import Scene.ShadingData;

/** Get material reflectance based on the metallic value.
 */
float3 getMaterialReflectanceForDeltaPaths(
    const MaterialType materialType,
    const bool hasDeltaLobes,
    const ShadingData sd,
    const BSDFProperties bsdfProperties
)
{
    if (materialType == MaterialType::Standard)
    {
        const BasicMaterialData md = gScene.materials.getBasicMaterialData(sd.materialID);
        const float metallic = md.specular.b; // Blue component stores metallic in MetalRough mode.

        if (metallic < 1.f)
        {
            // Use sum of reflection/transmission albedo as they are denoised together.
            const float3 diffuseReflectance =
                max(kNRDMinReflectance, bsdfProperties.diffuseReflectionAlbedo + bsdfProperties.diffuseTransmissionAlbedo);
            return diffuseReflectance;
        }
        // Handle only non-delta specular lobes.
        // We use an approximate shading model with a single specular lobe.
        else if (!hasDeltaLobes)
        {
            const float NdotV = saturate(dot(bsdfProperties.guideNormal, sd.V));
            const float ggxAlpha = bsdfProperties.roughness * bsdfProperties.roughness;
            float3 specularReflectance = approxSpecularIntegralGGX(bsdfProperties.specularReflectionAlbedo, ggxAlpha, NdotV);
            specularReflectance = max(kNRDMinReflectance, specularReflectance);
            return specularReflectance;
        }
    }
    else if (materialType == MaterialType::Hair)
    {
        // Use sum of reflection/transmission albedo as they are denoised together.
        const float3 reflectance =
            max(kNRDMinReflectance, bsdfProperties.diffuseReflectionAlbedo + bsdfProperties.diffuseTransmissionAlbedo);
        return reflectance;
    }

    return 1.f;
}

bool isDeltaReflectionAllowedAlongDeltaTransmissionPath(const ShadingData sd)
{
    const BasicMaterialData md = gScene.materials.getBasicMaterialData(sd.materialID);
    const float metallic = md.specular.b; // Blue component stores metallic in MetalRough mode.
    const float insideIoR = gScene.materials.evalIoR(sd.materialID);

    const float eta = sd.IoR / insideIoR;
    bool totalInternalReflection = evalFresnelDielectric(eta, sd.frame.toLocal(sd.V).z) == 1.f;

    bool nonTransmissiveMirror = (md.specularTransmission == 0.f) && (metallic == 1.f);

    return totalInternalReflection || nonTransmissiveMirror;
}
