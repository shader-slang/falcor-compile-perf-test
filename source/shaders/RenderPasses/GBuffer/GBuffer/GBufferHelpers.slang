/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/

/**
 * Helper functions for generating the G-buffer.
 */

import Scene.SceneTypes;
import Scene.ShadingData;
import Rendering.Materials.IMaterialInstance;
import Utils.Math.MathHelpers;
import Utils.Math.FormatConversion;
__exported import Scene.Material.ShadingUtils;

struct GBufferData
{
    float4 posW;
    float4 normW;
    float4 tangentW;
    float4 faceNormalW;
    float4 guideNormalW;
    float2 texC;
    uint4 mtlData;

    // Legacy channels
    float4 diffuseOpacity;
    float4 specRough;
    float4 emissive;
};

/**
 * Helper function to prepare G-buffer data.
 */
GBufferData prepareGBufferData(const ShadingData sd, const VertexData v, const IMaterialInstance mi, const BSDFProperties bsdfProperties)
{
    GBufferData gbuf = {};

    // We store the final normal in the G-buffer but pass along the original geometric tangent and its handedness (sign) unmodified.
    // This is needed for correctly orthonormalizing the tangent frame and computing the bitangent in passes that consume the G-buffer data.
    float bitangentSign = sd.frame.getHandednessSign();

    gbuf.posW = float4(sd.posW, 1.f);
    gbuf.normW = float4(sd.frame.N, 0.f);
    gbuf.tangentW = v.tangentW;
    gbuf.faceNormalW = float4(sd.faceN, 0.f);
    gbuf.guideNormalW = float4(bsdfProperties.guideNormal, 0.f);
    gbuf.texC = sd.uv;
    gbuf.mtlData = uint4(sd.materialID, sd.mtl.packedData.x, sd.mtl.packedData.y, mi.getLobeTypes(sd));

    // Setup legacy material channels.
    gbuf.diffuseOpacity = float4(bsdfProperties.diffuseReflectionAlbedo, 1.f);
    gbuf.specRough = float4(bsdfProperties.specularReflectance, bsdfProperties.roughness);
    gbuf.emissive = float4(bsdfProperties.emission, 0.f);

    return gbuf;
}
