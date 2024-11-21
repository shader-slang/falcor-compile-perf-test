/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"

import Scene.Shading;
import Utils.Sampling.UniformSampleGenerator;
import DiffRendering.SceneGradients;
import DiffRendering.GradientIOWrapper;

RWStructuredBuffer<float> materialGrad;
RWStructuredBuffer<float> geometryGrad;

cbuffer CB
{
    float3 gWi;
    float3 gWo;
}

[Differentiable]
IMaterialInstance getDiffMaterialInstance(out DiffMaterialData diffData, const ShadingData sd)
{
    let lod = ExplicitLodTextureSampler(0.f);
    let material = gScene.materials.getMaterial(sd.materialID);
    let mi = material.setupDiffMaterialInstance(diffData, gScene.materials, sd, lod);
    return mi;
}

[Differentiable]
float3 evalAD_PBRTDiffuse(uint2 pixelID)
{
    // Setup shading data struct.
    ShadingData sd = {};
    sd.frame = ShadingFrame::createIdentity();
    sd.faceN = detach(sd.frame.N);
    sd.frontFacing = true;

    uint materialID = 0;
    sd.mtl = gScene.materials.getMaterialHeader(materialID);
    sd.materialID = materialID;
    sd.IoR = 1.f;

    sd.V = gWi;

    sd.materialGradOffset = 0;
    sd.geometryGradOffset = 0;
    sd.threadID = 0;

    // Create MaterialInstance.
    DiffMaterialData diffData = DiffMaterialData();
    IMaterialInstance mi = getDiffMaterialInstance(diffData, sd);

    UniformSampleGenerator sg = UniformSampleGenerator(pixelID, 0);

    // Setup differentiable wo.
    uint hashIndex = hashFunction(sd.threadID, gSceneGradients.getHashSize(GradientType::MeshNormal));
    GradientIOWrapper gradIO = GradientIOWrapper(GradientType::MeshNormal, sd.geometryGradOffset, hashIndex);
    float3 wo = gradIO.getFloat(gWo, 0);

    float3 value = mi.evalAD(diffData, sd, wo, sg);
    return value;
}

[numthreads(1, 1, 1)]
void testDiffPBRTDiffuse(uint3 threadID: SV_DispatchThreadID)
{
    bwd_diff(evalAD_PBRTDiffuse)(threadID.xy, float3(1.f));

    [ForceUnroll]
    for (uint i = 0; i < 3; i++)
    {
        materialGrad[i] = gSceneGradients.getGrad(GradientType::Material, i, 0);
        geometryGrad[i] = gSceneGradients.getGrad(GradientType::MeshNormal, i, 0);
    }
}
