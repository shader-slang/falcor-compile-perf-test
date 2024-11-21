/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"

import Scene.Scene;

cbuffer CB
{
    uint gTriangleCount;                    ///< Total number of triangles.
}

StructuredBuffer<MeshLightData> gMeshData;                  ///< Per-mesh data for emissive meshes.
RWStructuredBuffer<PackedEmissiveTriangle> gTriangleData;   ///< Per-triangle geometry data for emissive triangles.

/** Kernel updating the emissive triangles for all mesh lights.
    Single dispatch with one thread per triangle.
*/
[numthreads(256, 1, 1)]
void updateTriangleVertices(uint3 DTid : SV_DispatchThreadID)
{
    const uint triIdx = DTid.x; // Global emissive triangle index.
    const uint vtxIdx = triIdx * 3; // The emissive triangle list is non-indexed.

    if (triIdx >= gTriangleCount) return;

    // Get the data for the mesh that this triangle belongs to.
    uint lightIdx = gTriangleData[triIdx].lightIdx;
    const MeshLightData meshData = gMeshData[lightIdx];

    GeometryInstanceID instanceID = { meshData.instanceID };
    uint triangleIndex = triIdx - meshData.triangleOffset; // Local triangle index in the mesh

    // Load triangle data.
    EmissiveTriangle tri = gTriangleData[triIdx].unpack();

    // Update triangle data.
    gScene.getVertexPositionsW(instanceID, triangleIndex, tri.posW);
    tri.normal = gScene.computeFaceNormalAndAreaW(instanceID, tri.posW, tri.area);
    gTriangleData[triIdx].pack(tri);
}
