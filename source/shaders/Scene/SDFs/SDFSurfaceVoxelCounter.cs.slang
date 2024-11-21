/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import Scene.SDFs.SDFVoxelCommon;

#define GROUP_WIDTH 4

cbuffer CB
{
    uint gGridWidth;
};

Texture3D<float> gSDFGrid;
RWByteAddressBuffer gTotalVoxelCount;

groupshared uint gGroupVoxelCount;
groupshared float gGroupValues[(GROUP_WIDTH + 1) * (GROUP_WIDTH + 1) * (GROUP_WIDTH + 1)];

uint getGroupValueIndex(uint3 groupValueCoords)
{
    return groupValueCoords.x + (GROUP_WIDTH + 1) * (groupValueCoords.y + (GROUP_WIDTH + 1) * groupValueCoords.z);
}

[numthreads(GROUP_WIDTH, GROUP_WIDTH, GROUP_WIDTH)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID)
{
    if (any(dispatchThreadID >= gGridWidth)) return;

    const uint3 voxelCoords = dispatchThreadID;
    const bool isGroupManager = all(groupThreadID == uint3(0));

    // Initialize group values.
    if (isGroupManager)
    {
        gGroupVoxelCount = 0;
    }

    gGroupValues[getGroupValueIndex(groupThreadID)] = gSDFGrid[voxelCoords].x;

    bool3 borderThread = groupThreadID == GROUP_WIDTH - 1 || dispatchThreadID == gGridWidth - 1;

    if (borderThread.x)
    {
        gGroupValues[getGroupValueIndex(groupThreadID + uint3(1, 0, 0))] = gSDFGrid[voxelCoords + uint3(1, 0, 0)].x;

        if (borderThread.y)
        {
            gGroupValues[getGroupValueIndex(groupThreadID + uint3(0, 1, 0))] = gSDFGrid[voxelCoords + uint3(0, 1, 0)].x;
            gGroupValues[getGroupValueIndex(groupThreadID + uint3(1, 1, 0))] = gSDFGrid[voxelCoords + uint3(1, 1, 0)].x;

            if (borderThread.z)
            {
                gGroupValues[getGroupValueIndex(groupThreadID + uint3(0, 0, 1))] = gSDFGrid[voxelCoords + uint3(0, 0, 1)].x;
                gGroupValues[getGroupValueIndex(groupThreadID + uint3(0, 1, 1))] = gSDFGrid[voxelCoords + uint3(0, 1, 1)].x;
                gGroupValues[getGroupValueIndex(groupThreadID + uint3(1, 0, 1))] = gSDFGrid[voxelCoords + uint3(1, 0, 1)].x;
                gGroupValues[getGroupValueIndex(groupThreadID + uint3(1, 1, 1))] = gSDFGrid[voxelCoords + uint3(1, 1, 1)].x;
            }
        }
        else if (borderThread.z)
        {
            gGroupValues[getGroupValueIndex(groupThreadID + uint3(0, 0, 1))] = gSDFGrid[voxelCoords + uint3(0, 0, 1)].x;
            gGroupValues[getGroupValueIndex(groupThreadID + uint3(1, 0, 1))] = gSDFGrid[voxelCoords + uint3(1, 0, 1)].x;
        }
    }
    else if (borderThread.y)
    {
        gGroupValues[getGroupValueIndex(groupThreadID + uint3(0, 1, 0))] = gSDFGrid[voxelCoords + uint3(0, 1, 0)].x;

        if (borderThread.z)
        {
            gGroupValues[getGroupValueIndex(groupThreadID + uint3(0, 0, 1))] = gSDFGrid[voxelCoords + uint3(0, 0, 1)].x;
            gGroupValues[getGroupValueIndex(groupThreadID + uint3(0, 1, 1))] = gSDFGrid[voxelCoords + uint3(0, 1, 1)].x;
        }
    }
    else if (borderThread.z)
    {
        gGroupValues[getGroupValueIndex(groupThreadID + uint3(0, 0, 1))] = gSDFGrid[voxelCoords + uint3(0, 0, 1)].x;
    }

    GroupMemoryBarrierWithGroupSync();
    
    // Load voxel corner values.
    float4 values0xx;
    float4 values1xx;
    values0xx[0] = gGroupValues[getGroupValueIndex(groupThreadID)];
    values0xx[1] = gGroupValues[getGroupValueIndex(groupThreadID + uint3(0, 0, 1))];
    values0xx[2] = gGroupValues[getGroupValueIndex(groupThreadID + uint3(0, 1, 0))];
    values0xx[3] = gGroupValues[getGroupValueIndex(groupThreadID + uint3(0, 1, 1))];
    values1xx[0] = gGroupValues[getGroupValueIndex(groupThreadID + uint3(1, 0, 0))];
    values1xx[1] = gGroupValues[getGroupValueIndex(groupThreadID + uint3(1, 0, 1))];
    values1xx[2] = gGroupValues[getGroupValueIndex(groupThreadID + uint3(1, 1, 0))];
    values1xx[3] = gGroupValues[getGroupValueIndex(groupThreadID + uint3(1, 1, 1))];

    if (SDFVoxelCommon::containsSurface(values0xx, values1xx))
    {
        InterlockedAdd(gGroupVoxelCount, 1);
    }

    GroupMemoryBarrierWithGroupSync();

    if (isGroupManager)
    {
        gTotalVoxelCount.InterlockedAdd(0, gGroupVoxelCount);
    }
}
