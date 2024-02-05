/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
#include "Scene/SceneDefines.slangh"

__exported import Scene.Shading;
import Scene.Material.ShadingUtils;
import Rendering.Materials.TexLODHelpers;
import Utils.Math.MathHelpers;
import Internal.Utils.Geometry.GeometryHelpersInternal;
import Params;

/** Compute derivatives at primary triangle hit.
*/
void computeDerivativesAtPrimaryTriangleHit(const TriangleHit hit, uint2 pixel, uint2 frameDim, out float2 ddx, out float2 ddy)
{
    // Get world space position
    VertexData v = gScene.getVertexData(hit);
    float hitT = length(v.posW - gScene.camera.getPosition());
    const float3 nonNormalizedRayDir = gScene.camera.computeNonNormalizedRayDirPinhole(pixel, frameDim);
    float3 rayDir = normalize(nonNormalizedRayDir);

    float3 dDdx, dDdy;
    computeRayDirectionDifferentials(nonNormalizedRayDir, gScene.camera.data.cameraU, gScene.camera.data.cameraV, frameDim, dDdx, dDdy);

    RayDiff rd = RayDiff(float3(0.f), float3(0.f), dDdx, dDdy);
    rd = rd.propagate(v.posW.xyz, rayDir, hitT, v.faceNormalW);

    // Bounce using primary hit geometry.
    const float4x4 worldMat = gScene.getWorldMatrix(hit.instanceID);
    const float3x3 worldInvTransposeMat = gScene.getInverseTransposeWorldMatrix(hit.instanceID);
    const uint3 vertexIndices = gScene.getIndices(hit.instanceID, hit.primitiveIndex);
    StaticVertexData vertices[3] = { gScene.getVertex(vertexIndices[0]), gScene.getVertex(vertexIndices[1]), gScene.getVertex(vertexIndices[2]) };
    float2 dBarydx, dBarydy;
    float3 unnormalizedN, normals[3];

    prepareRayDiffAtHitPoint(v, vertices, hit.getBarycentricWeights(), rayDir, worldMat, worldInvTransposeMat, rd, unnormalizedN, normals, dBarydx, dBarydy, ddx, ddy);
}

/** Setup vertex data based on hit point.
    \param[in] hit Hit information.
    \param[in] rayOrigin Ray origin.
    \param[in] rayDir Normalized ray direction.
    \param[out] materialID Material ID.
    \return VertexData struct.
*/
VertexData loadVertexData(const HitInfo hit, const float3 rayOrigin, const float3 rayDir, out uint materialID)
{
    VertexData v = {};
    materialID = {};

#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_TRIANGLE_MESH)
    if (hit.getType() == HitType::Triangle)
    {
        const TriangleHit triangleHit = hit.getTriangleHit();
        StaticVertexData vertices[3];
        v = gScene.getVertexData(triangleHit, vertices);
        materialID = gScene.getMaterialID(triangleHit.instanceID);

        bool isFrontFaceCW = gScene.isObjectFrontFaceCW(triangleHit.instanceID);
        float4x4 worldMat = gScene.getWorldMatrix(triangleHit.instanceID);
        float3x3 worldInvTransposeMat = gScene.getInverseTransposeWorldMatrix(triangleHit.instanceID);
        v.normalCurvature = computeNormalCurvature(vertices, triangleHit.getBarycentricWeights(), isFrontFaceCW, worldMat, worldInvTransposeMat, -rayDir);
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_DISPLACED_TRIANGLE_MESH)
    if (hit.getType() == HitType::DisplacedTriangle)
    {
        const DisplacedTriangleHit displacedTriangleHit = hit.getDisplacedTriangleHit();
        v = gScene.getVertexData(displacedTriangleHit, -rayDir);
        materialID = gScene.getMaterialID(displacedTriangleHit.instanceID);
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_CURVE)
    if (hit.getType() == HitType::Curve)
    {
        const CurveHit curveHit = hit.getCurveHit();
        v = gScene.getVertexDataFromCurve(curveHit);
        materialID = gScene.getMaterialID(curveHit.instanceID);
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_CURVE_OTS)
    if (hit.getType() == HitType::CurveOTS)
    {
        const CurveOTSHit curveOTSHit = hit.getCurveOTSHit();
        v = gScene.getVertexData(curveOTSHit, rayOrigin);
        materialID = gScene.getMaterialID(curveOTSHit.instanceID);
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_SDF_GRID)
    if (hit.getType() == HitType::SDFGrid)
    {
        const SDFGridHit sdfGridHit = hit.getSDFGridHit();
#ifdef FALCOR_INTERNAL
        v = gScene.getVertexDataFromSDFGrid(sdfGridHit);
#else
        v = gScene.getVertexDataFromSDFGrid(sdfGridHit, rayOrigin, rayDir);
#endif
        materialID = gScene.getMaterialID(sdfGridHit.instanceID);
    }
#endif
    return v;
}

/** Determine hints to use when creating the material instance.
*/
uint getMaterialInstanceHints(const HitInfo hit, const bool isPrimary)
{
    uint hints = 0;
    if (hit.getType() == HitType::Triangle || hit.getType() == HitType::DisplacedTriangle)
    {
#if GBUFFER_ADJUST_SHADING_NORMALS
        if (kAdjustShadingNormals || isPrimary) hints |= (uint)MaterialInstanceHints::AdjustShadingNormal;
#else
        if (kAdjustShadingNormals && !isPrimary) hints |= (uint)MaterialInstanceHints::AdjustShadingNormal;
#endif
    }
    return hints;
}

/** Setup ShadingData based on loaded vertex/material attributes for a hit point.
    \param[in] hit Hit information.
    \param[in] rayOrigin Ray origin.
    \param[in] rayDir Normalized ray direction.
    \param[in] lod Method for computing texture level-of-detail.
    \return ShadingData struct.
*/
ShadingData loadShadingData(const HitInfo hit, const float3 rayOrigin, const float3 rayDir, const ITextureSampler lod)
{
    uint materialID = {};

    VertexData v = loadVertexData(hit, rayOrigin, rayDir, materialID);
    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -rayDir, lod);
    sd.setHitInfo(hit);
    return sd;
}
