/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"
import Scene.SDFs.SDFVoxelCommon;
import Utils.Math.PackedFormats;
import Utils.Math.IntervalArithmetic;

static const uint kGroupWidth = GROUP_WIDTH;
static const uint kBrickWidth = BRICK_WIDTH;
static const uint kGroupSize = GROUP_WIDTH * GROUP_WIDTH * GROUP_WIDTH;
static const uint kChunkWidth = CHUNK_WIDTH;

struct ParamBlock
{
    uint virtualGridWidth;
    Texture3D<float> sdfGrid;
    RWTexture3D<float2> intervalDistances;
    Texture3D<float2> parentIntervalDistances;
};

ParameterBlock<ParamBlock> gParamBlock;

#if GROUP_BRICK_CREATION
groupshared float2 gGroupIntervalValues[kGroupSize];
#endif

float2 fetchCubeInterval(uint3 voxelCoords)
{
    // Gather min and max distances from the specified voxel.
    float4 values0xx;
    float4 values1xx;
    values0xx[0] = gParamBlock.sdfGrid[voxelCoords].x;
    values0xx[1] = gParamBlock.sdfGrid[voxelCoords + uint3(0, 0, 1)].x;
    values0xx[2] = gParamBlock.sdfGrid[voxelCoords + uint3(0, 1, 0)].x;
    values0xx[3] = gParamBlock.sdfGrid[voxelCoords + uint3(0, 1, 1)].x;
    values1xx[0] = gParamBlock.sdfGrid[voxelCoords + uint3(1, 0, 0)].x;
    values1xx[1] = gParamBlock.sdfGrid[voxelCoords + uint3(1, 0, 1)].x;
    values1xx[2] = gParamBlock.sdfGrid[voxelCoords + uint3(1, 1, 0)].x;
    values1xx[3] = gParamBlock.sdfGrid[voxelCoords + uint3(1, 1, 1)].x;

    float4 minValues = min(values0xx, values1xx);
    float4 maxValues = max(values0xx, values1xx);
    float minDist = min(minValues.x, min(minValues.y, min(minValues.z, minValues.w)));
    float maxDist = max(maxValues.x, max(maxValues.y, max(maxValues.z, maxValues.w)));

    return float2(minDist, maxDist);
}

[numthreads(kGroupWidth, kGroupWidth, kGroupWidth)]
void rootGather(uint3 dispatchThreadID : SV_DispatchThreadID, uint groupThreadIdx : SV_GroupIndex, uint3 groupID : SV_GroupID)
{
    if (any(dispatchThreadID >= gParamBlock.virtualGridWidth))
        return;

    const uint3 voxelCoords = dispatchThreadID;

#if GROUP_BRICK_CREATION
    const bool isGroupManager = groupThreadIdx == 0;

    // Result is in the format (min distance, max distance)
    float2 result = fetchCubeInterval(voxelCoords);
    gGroupIntervalValues[groupThreadIdx] = result;

    GroupMemoryBarrierWithGroupSync();

    if (isGroupManager)
    {
        // Gather all min and max distances from all voxels in one block (in this case a thread group).
        for (uint v = 0; v < kGroupSize; v++)
        {
            result = ivlCombine(result, gGroupIntervalValues[v]);
        }

        gParamBlock.intervalDistances[groupID] = result;
    }
#else
    // Result is in format (min value, max value)
    float2 result = float2(FLT_MAX, -FLT_MAX);

    uint3 parentCoords = voxelCoords * kBrickWidth;
    for (uint x = 0; x < kBrickWidth; x++)
    {
        for (uint y = 0; y < kBrickWidth; y++)
        {
            for (uint z = 0; z < kBrickWidth; z++)
            {
                result = ivlCombine(result, fetchCubeInterval(parentCoords + uint3(x, y, z)));
            }
        }
    }

    gParamBlock.intervalDistances[voxelCoords] = result;
#endif
}

[numthreads(kGroupWidth, kGroupWidth, kGroupWidth)]
void chunkGather(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (any(dispatchThreadID >= gParamBlock.virtualGridWidth))
        return;

    const uint3 voxelCoords = dispatchThreadID;

    // Result is in format (min value, max value)
    float2 result = float2(FLT_MAX, -FLT_MAX);

    // Load voxel corner values.
    uint3 parentCoords = voxelCoords * kChunkWidth;
    for (uint x = 0; x < kChunkWidth; x++)
    {
        for (uint y = 0; y < kChunkWidth; y++)
        {
            for (uint z = 0; z < kChunkWidth; z++)
            {
                result = ivlCombine(result, gParamBlock.parentIntervalDistances[parentCoords + uint3(x, y, z)]);
            }
        }
    }

    gParamBlock.intervalDistances[voxelCoords] = result;
}
