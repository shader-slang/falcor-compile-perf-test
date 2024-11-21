/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import Scene.SceneTypes;
import Utils.Math.AABB;

/** Compute pass for updating AABBs of animated vertices.

    The dispatch size is one thread per AABB.
*/

struct CurveAABBUpdater
{
    uint AABBCount;
    uint dimX;

    // Curve data
    ByteAddressBuffer curveIndexData;
    StructuredBuffer<StaticCurveVertexData> curveVertices;
    RWStructuredBuffer<AABB> curveAABBs;

    void updateAABB(uint3 dispatchThreadID)
    {
        uint indexID = dispatchThreadID.y * dimX + dispatchThreadID.x;
        if (indexID >= AABBCount) return;

        uint vertexID = curveIndexData.Load(indexID * 4);

        AABB bbox;
        bbox.invalidate();
        for (uint i = 0; i < 2; i++)
        {
            const StaticCurveVertexData vertex = curveVertices[vertexID + i];
            bbox.include(vertex.position - vertex.radius);
            bbox.include(vertex.position + vertex.radius);
        }

        curveAABBs[indexID] = bbox;
    }
};

ParameterBlock<CurveAABBUpdater> gCurveAABBUpdater;

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    gCurveAABBUpdater.updateAABB(dispatchThreadID);
}
