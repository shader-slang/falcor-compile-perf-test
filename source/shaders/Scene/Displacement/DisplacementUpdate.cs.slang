/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import Scene.Scene;
import Scene.Displacement.DisplacementMapping;
import Utils.Math.AABB;
import DisplacementUpdateTask;

static const bool kUsePreciseShellBounds = true;

cbuffer CB
{
    uint gTaskCount;

    StructuredBuffer<DisplacementUpdateTask> gTasks;
    RWStructuredBuffer<AABB> gAABBs;
};

/** This kernel is used for computing AABBs for displaced triangles.
    Work is organized in tasks (described by DisplacementUpdateTask).
    Each tasks computes AABBs for a range of triangles from a single mesh.
    A fixed number of threads (DisplacementUpdateTask::kThreadCount) is launched for each task,
    processing triangles in a fixed stride of kThreadCount.
*/
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint threadIndex = dispatchThreadId.x;
    const uint taskIndex = dispatchThreadId.y;

    if (threadIndex >= DisplacementUpdateTask::kThreadCount || taskIndex >= gTaskCount) return;

    const DisplacementUpdateTask task = gTasks[taskIndex];

    const uint materialID = gScene.meshes[task.meshID].materialID;

    DisplacementData displacementData;
    gScene.materials.loadDisplacementData(materialID, displacementData);

    const float globalExpansion = displacementData.getConservativeGlobalExpansion();
    const uint iterationCount = (task.count + DisplacementUpdateTask::kThreadCount - 1) / DisplacementUpdateTask::kThreadCount;

    for (uint iteration = 0; iteration < iterationCount; ++iteration)
    {
        const uint index = iteration * DisplacementUpdateTask::kThreadCount + threadIndex;
        if (index >= task.count) return;

        const uint triangleIndex = task.triangleIndex + index;
        const uint AABBIndex = task.AABBIndex + index;

        const uint3 indices = gScene.getIndices(task.meshID, triangleIndex);
        StaticVertexData vertices[3] = { gScene.getVertex(indices[0]), gScene.getVertex(indices[1]), gScene.getVertex(indices[2]) };

        AABB aabb;
        aabb.invalidate();

        if (!kUsePreciseShellBounds)
        {
            for (uint i = 0; i < 3; ++i) aabb.include(vertices[i].position);

            aabb.minPoint -= globalExpansion;
            aabb.maxPoint += globalExpansion;
        }
        else
        {
            const float2 shellMinMax = displacementData.getShellMinMax(vertices[0].texCrd, vertices[1].texCrd, vertices[2].texCrd);

            for (uint i = 0; i < 3; ++i)
            {
                aabb.include(vertices[i].position + (vertices[i].normal) * shellMinMax.x);
                aabb.include(vertices[i].position + (vertices[i].normal) * shellMinMax.y);

            }
        }

        gAABBs[AABBIndex] = aabb;
    }
}
