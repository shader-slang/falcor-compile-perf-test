/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"

import Scene.Scene;
import Utils.Color.ColorHelpers;

cbuffer CB
{
    uint gLightIdx;                         ///< The mesh light index.
    uint gMaterialID;                       ///< The material ID of the mesh light.
    uint gInstanceID;                       ///< Mesh light's global geometry instance ID. We use this to find its data in the scene.
    uint gTriangleCount;                    ///< Number of triangles in current mesh light.
    uint gTriangleOffset;                   ///< Current offset into global list of emissive triangles.
}

RWStructuredBuffer<PackedEmissiveTriangle> gTriangleData;   ///< Per-triangle data for emissive triangles.

/** Kernel building the emissive triangles list for all mesh lights.
    One dispatch per mesh light with one thread per triangle.
*/
[numthreads(256, 1, 1)]
void buildTriangleList(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= gTriangleCount) return;

    const GeometryInstanceID instanceID = { gInstanceID };
    const uint triangleIndex = DTid.x; // Triangle index in current mesh.
    const uint triIdx = gTriangleOffset + triangleIndex;
    const uint vtxIdx = triIdx * 3; // The emissive triangle list is non-indexed.

    // Fetch vertex data.
    EmissiveTriangle tri;
    gScene.getVertexPositionsW(instanceID, triangleIndex, tri.posW);
    gScene.getVertexTexCoords(instanceID, triangleIndex, tri.texCoords);
    tri.normal = gScene.computeFaceNormalAndAreaW(instanceID, tri.posW, tri.area);;
    tri.materialID = gMaterialID;
    tri.lightIdx = gLightIdx;

    // Store triangle data.
    gTriangleData[triIdx].pack(tri);
}
