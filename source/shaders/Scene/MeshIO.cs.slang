/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import Scene.SceneTypes;
import Scene.Scene;

struct MeshLoader
{
    uint vertexCount;
    uint vbOffset;
    uint triangleCount;
    uint ibOffset;
    bool use16BitIndices;

    ParameterBlock<Scene> scene;

    // Output
    RWStructuredBuffer<float3> positions;
    RWStructuredBuffer<float3> texcrds;
    RWStructuredBuffer<uint3> triangleIndices;

    void getMeshIndices(uint triangleId)
    {
        if (triangleId >= triangleCount) return;
        uint3 vtxIndices = scene.getLocalIndices(ibOffset, triangleId, use16BitIndices);
        triangleIndices[triangleId] = vtxIndices;
    }

    void getMeshVertexData(uint vertexId)
    {
        if (vertexId >= vertexCount) return;
        StaticVertexData vtxData = scene.getVertex(vertexId + vbOffset);
        positions[vertexId] = vtxData.position;
        texcrds[vertexId] = float3(vtxData.texCrd, 0.f);
    }
};

struct MeshUpdater
{
    uint vertexCount;
    uint vbOffset;

    StructuredBuffer<float3> positions;
    StructuredBuffer<float3> normals;
    StructuredBuffer<float3> tangents;
    StructuredBuffer<float3> texcrds;

    // Output
    RWStructuredBuffer<PackedStaticVertexData> vertexData;

    void setMeshVertexData(uint vertexId)
    {
        if (vertexId >= vertexCount) return;
        StaticVertexData vtxData;
        vtxData.position = positions[vertexId];
        vtxData.normal = normals[vertexId];
        vtxData.tangent = float4(tangents[vertexId], 1.f); // Tangent follows the orientation such that `b = cross(n, t)`.
        vtxData.texCrd = texcrds[vertexId].xy;
        vertexData[vertexId + vbOffset].pack(vtxData);
    }
};

ParameterBlock<MeshLoader> meshLoader;
ParameterBlock<MeshUpdater> meshUpdater;

[numthreads(256, 1, 1)]
void getMeshVerticesAndIndices(uint3 tid: SV_DispatchThreadID)
{
    meshLoader.getMeshIndices(tid.x);
    meshLoader.getMeshVertexData(tid.x);
}

[numthreads(256, 1, 1)]
void setMeshVertices(uint3 tid: SV_DispatchThreadID)
{
    meshUpdater.setMeshVertexData(tid.x);
}
