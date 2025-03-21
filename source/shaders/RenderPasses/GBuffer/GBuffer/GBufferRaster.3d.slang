/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#include "Scene/VertexAttrib.slangh"

import Scene.Raster;
import Utils.Math.MathHelpers;
import GBufferHelpers;
import Rendering.Materials.TexLODHelpers;

cbuffer PerFrameCB
{
    uint2 gFrameDim;
};

// GBuffer channels
struct GBufferPSOut
{
    float4 posW : SV_TARGET0;
    float4 normW : SV_TARGET1;
    float4 tangentW : SV_TARGET2;
    float4 faceNormalW : SV_TARGET3;
    float2 texC : SV_TARGET4;
    float4 texGrads : SV_TARGET5;
    float2 mvec : SV_TARGET6;
    uint4 mtlData : SV_TARGET7;
};

// GBufferRaster channels
RasterizerOrderedTexture2D<PackedHitInfo> gVBuffer;
RasterizerOrderedTexture2D<float4> gGuideNormalW;
RasterizerOrderedTexture2D<float4> gDiffOpacity;
RasterizerOrderedTexture2D<float4> gSpecRough;
RasterizerOrderedTexture2D<float4> gEmissive;
RasterizerOrderedTexture2D<float4> gViewW;
RasterizerOrderedTexture2D<float2> gPosNormalFwidth;
RasterizerOrderedTexture2D<float2> gLinearZAndDeriv;
RasterizerOrderedTexture2D<float> gMask;

#define is_valid(name) (is_valid_##name != 0)

VSOut vsMain(VSIn vsIn)
{
    return defaultVS(vsIn);
}

float2 computeMotionVector(const VSOut vsOut, const int2 ipos)
{
    // Current sample in pixel coords.
    float2 pixelPos = ipos + float2(0.5, 0.5);
    // Sample in previous frame in clip space coords, no jittering applied.
    float4 prevPosH = vsOut.prevPosH;
    // Remove camera jitter from motion vector
    return calcMotionVector(pixelPos, prevPosH, gFrameDim) + float2(gScene.camera.data.jitterX, -gScene.camera.data.jitterY);
}

[earlydepthstencil]
GBufferPSOut psMain(VSOut vsOut, uint triangleIndex: SV_PrimitiveID, float3 barycentrics: SV_Barycentrics)
{
    // Using vOut.posH.xy as pixel coordinate since it has the SV_Position semantic.
    int2 ipos = int2(vsOut.posH.xy);

    float3 faceNormal = gScene.getFaceNormalW(vsOut.instanceID, triangleIndex);
    VertexData v = prepareVertexData(vsOut, faceNormal);
    let lod = ImplicitLodTextureSampler();

#if USE_ALPHA_TEST
    if (gScene.materials.alphaTest(v, vsOut.materialID, lod))
        discard;
#endif
    const float3 viewDir = normalize(gScene.camera.getPosition() - v.posW);
    ShadingData sd = gScene.materials.prepareShadingData(v, vsOut.materialID, viewDir, lod);

    uint hints = 0;
#if ADJUST_SHADING_NORMALS
    hints |= (uint)MaterialInstanceHints::AdjustShadingNormal;
#endif

    // Create material instance and query its properties.
    let mi = gScene.materials.getMaterialInstance(sd, lod, hints);
    let bsdfProperties = mi.getProperties(sd);

    const GBufferData gbuf = prepareGBufferData(sd, v, mi, bsdfProperties);

    const float4 texGrads = float4(ddx(sd.uv), ddy(sd.uv));
    const float2 mvec = computeMotionVector(vsOut, ipos);

    GBufferPSOut psOut = {};

    // Store render target outputs.
    if (is_valid(gPosW))
        psOut.posW = gbuf.posW;
    if (is_valid(gNormW))
        psOut.normW = gbuf.normW;
    if (is_valid(gTangentW))
        psOut.tangentW = gbuf.tangentW;
    if (is_valid(gFaceNormalW))
        psOut.faceNormalW = gbuf.faceNormalW;
    if (is_valid(gTexC))
        psOut.texC = gbuf.texC;
    if (is_valid(gTexGrads))
        psOut.texGrads = texGrads;
    if (is_valid(gMotionVector))
        psOut.mvec = mvec;
    if (is_valid(gMaterialData))
        psOut.mtlData = gbuf.mtlData;

    // Store UAV outputs.
    if (is_valid(gViewW))
        gViewW[ipos] = float4(sd.V, 0.f);
    if (is_valid(gGuideNormalW))
        gGuideNormalW[ipos] = gbuf.guideNormalW;
    if (is_valid(gDiffOpacity))
        gDiffOpacity[ipos] = gbuf.diffuseOpacity;
    if (is_valid(gSpecRough))
        gSpecRough[ipos] = gbuf.specRough;
    if (is_valid(gEmissive))
        gEmissive[ipos] = gbuf.emissive;
    if (is_valid(gMask))
        gMask[ipos] = 1.f;

    // Length of derivatives of position and guide normal.
    // This is a feature guide for the SVGF denoiser pass.
    if (is_valid(gPosNormalFwidth))
    {
        gPosNormalFwidth[ipos] = float2(length(fwidth(sd.posW)), length(fwidth(bsdfProperties.guideNormal)));
    }

    // Linear z and its derivative
    if (is_valid(gLinearZAndDeriv))
    {
        const float linearZ = vsOut.posH.z * vsOut.posH.w;
        gLinearZAndDeriv[ipos] = float2(linearZ, max(abs(ddx(linearZ)), abs(ddy(linearZ))));
    }

    // Note on barycentrics:
    // The barycentric weights provided to pixel shader correspond to vertices A, B, C of the rasterized triangle.
    // For triangle strips, every odd primitive has the order for vertices B and C flipped. We don't handle triangle
    // strips as DXR does not support them and Falcor uses triangle lists exclusively in its scene representation.
    // DXR intersection attributes store barycentric weights in a float2 for triangle vertices B and C.
    // This is what we store in the hit info.

    // Store hit information.
    if (is_valid(gVBuffer))
    {
        TriangleHit triangleHit;
        triangleHit.instanceID = vsOut.instanceID;
        triangleHit.primitiveIndex = triangleIndex;
        triangleHit.barycentrics = barycentrics.yz;
        gVBuffer[ipos] = triangleHit.pack();
    }

    return psOut;
}
