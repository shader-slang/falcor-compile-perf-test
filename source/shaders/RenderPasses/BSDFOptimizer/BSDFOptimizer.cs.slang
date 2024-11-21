/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"

import Scene.Shading;
import Scene.Material.ShadingUtils;
import Utils.Sampling.SampleGenerator;
import Utils.Math.BitTricks;
import Utils.Math.MathHelpers;
import BSDFOptimizerParams;
import BSDFOptimizerHelpers;

cbuffer CB
{
    BSDFOptimizerParams params;
};

struct SurfaceData : IDifferentiable
{
    ShadingData sd;
    float3 wo;
};

[Differentiable]
SurfaceData prepareShadingData(const VertexData v, const float3 viewDir, const uint materialID, const ITextureSampler lod)
{
    SurfaceData data = {};

    // Setup Falcor's ShadingData based on the selected scene material and lobes.
    data.sd = no_diff gScene.materials.prepareShadingData(v, materialID, viewDir, lod);
    return data;
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
float3 evalBSDFSlice(bool isRef, const uint2 pixel)
{
    float2 uv = no_diff getViewportCoord(pixel, float2(0.f), params.viewPortScale);
    let lod = ExplicitLodTextureSampler(0.f);
    SampleGenerator sg = SampleGenerator(pixel, params.frameCount);

    // Calculate geometry and incident/outgoing directions.
    VertexData v;
    float3 viewDir;
    float3 lightDir = no_diff calculateSliceGeometry(uv, v, viewDir);

    // Setup shading data.
    SurfaceData data = prepareShadingData(v, viewDir, isRef ? params.refMaterialID : params.initMaterialID, lod);
    data.wo = detach(lightDir);

    // Set offset for writing gradients.
    data.sd.materialGradOffset = 0;
    data.sd.threadID = 0;

    // Create differentiable BSDF instance and evaluate BSDF at shading point.
    DiffMaterialData diffData = DiffMaterialData();
    let mi = getDiffMaterialInstance(diffData, data.sd);
    float3 f = isRef ? no_diff mi.eval(data.sd, data.wo, sg) : mi.evalAD(diffData, data.sd, data.wo, sg);

    // Remove cosine term.
    float NdotL = abs(dot(data.sd.frame.N, data.wo));
    f = NdotL > 0.f ? f / NdotL : float3(0);

    return f;
}

[Differentiable]
float3 computeLoss(const uint2 pixel)
{
    float3 fRef = evalBSDFSlice(true, pixel);
    float3 fCur = evalBSDFSlice(false, pixel);

    float3 diff = fCur - fRef;
    return 0.5 * diff * diff;
}

void execute(const uint2 pixel)
{
    if (any(pixel >= params.bsdfTableDim))
        return;
    bwd_diff(computeLoss)(pixel, float3(1.f));
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    execute(dispatchThreadID.xy);
}
