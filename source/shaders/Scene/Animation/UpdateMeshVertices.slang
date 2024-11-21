/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import Scene.SceneTypes;
import SharedTypes;

struct MeshPerKeyframe
{
    StructuredBuffer<PackedStaticVertexData> vertexData;
};

struct MeshVertexUpdater
{
    bool copyPrev;

    // Per Mesh
    StructuredBuffer<InterpolationInfo> perMeshInterp;
    StructuredBuffer<PerMeshMetadata> perMeshData;

    // All keyframes for all meshes in a linear array
#if MESH_KEYFRAME_COUNT > 0
    MeshPerKeyframe meshPerKeyframe[MESH_KEYFRAME_COUNT];
#else
    MeshPerKeyframe meshPerKeyframe[1];
#endif

    // Output
    RWStructuredBuffer<PackedStaticVertexData> sceneVertexData;
    RWStructuredBuffer<PrevVertexData> prevVertexData;

    StaticVertexData interpolateVertex(StaticVertexData v0, StaticVertexData v1, float t)
    {
        StaticVertexData out;
        out.position = lerp(v0.position, v1.position, t);
        out.normal = normalize(lerp(v0.normal, v1.normal, t));
        out.tangent = lerp(v0.tangent, v1.tangent, t);
        out.tangent.xyz = normalize(out.tangent.xyz);
        return out;
    }

    void updateVertex(uint meshVertexID, uint meshID)
    {
        PerMeshMetadata meta = perMeshData[meshID];
        if (meshVertexID >= meta.vertexCount) return;

        uint outVertexID = meta.sceneVbOffset + meshVertexID;
        StaticVertexData orig = sceneVertexData[outVertexID].unpack();
        uint prevVertexID = meta.prevVbOffset + meshVertexID;

        if (copyPrev)
        {
            prevVertexData[prevVertexID].position = orig.position;
            return;
        }

        InterpolationInfo interp = perMeshInterp[meshID];
        MeshPerKeyframe meshKeyframeA = meshPerKeyframe[meta.keyframeBufferOffset + interp.keyframeIndices.x];
        MeshPerKeyframe meshKeyframeB = meshPerKeyframe[meta.keyframeBufferOffset + interp.keyframeIndices.y];
        StaticVertexData v0 = meshKeyframeA.vertexData[meshVertexID].unpack();
        StaticVertexData v1 = meshKeyframeB.vertexData[meshVertexID].unpack();
        StaticVertexData result = interpolateVertex(v0, v1, interp.t);

        prevVertexData[prevVertexID].position = orig.position;

        result.texCrd = orig.texCrd; // Same
        sceneVertexData[outVertexID].pack(result);
    }
};

ParameterBlock<MeshVertexUpdater> gMeshVertexUpdater;

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    gMeshVertexUpdater.updateVertex(dispatchThreadID.x, dispatchThreadID.y);
}
