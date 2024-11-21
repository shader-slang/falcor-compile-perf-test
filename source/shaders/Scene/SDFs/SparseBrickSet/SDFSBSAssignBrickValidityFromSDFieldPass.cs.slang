/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import Scene.SDFs.SDFVoxelCommon;

static const uint kGroupWidth = GROUP_WIDTH;

struct ParamBlock
{
    uint virtualGridWidth;
    uint virtualBricksPerAxis;
    uint brickWidthInVoxels;
    Texture3D<float> sdfGrid;
    RWByteAddressBuffer brickValidity;
};

ParameterBlock<ParamBlock> gParamBlock;

#if GROUP_BRICK_CREATION
groupshared uint gBrickValid;
#endif

[numthreads(kGroupWidth, kGroupWidth, kGroupWidth)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID)
{
    if (any(dispatchThreadID >= gParamBlock.virtualGridWidth))
        return;

    const uint3 voxelCoords = dispatchThreadID;

    // Initialize group values.
#if GROUP_BRICK_CREATION
    const bool isGroupManager = all(groupThreadID == uint3(0));

    if (isGroupManager)
    {
        gBrickValid = 0;
    }

    GroupMemoryBarrierWithGroupSync();
#else
    uint voxelValid = 0;
#endif

    // Load voxel corner values.
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

    if (SDFVoxelCommon.containsSurface(values0xx, values1xx))
    {
#if GROUP_BRICK_CREATION
        gBrickValid = 1;
#else
        voxelValid = 1;
#endif
    }

#if GROUP_BRICK_CREATION
    GroupMemoryBarrierWithGroupSync();
#endif

#if GROUP_BRICK_CREATION
    if (isGroupManager)
    {
        uint3 virtualBrickCoords = voxelCoords / gParamBlock.brickWidthInVoxels;
        uint virtualBrickID = virtualBrickCoords.x + gParamBlock.virtualBricksPerAxis * (virtualBrickCoords.y + gParamBlock.virtualBricksPerAxis * virtualBrickCoords.z);

        gParamBlock.brickValidity.Store(virtualBrickID * 4, gBrickValid);
    }
#else
    uint3 virtualBrickCoords = voxelCoords / gParamBlock.brickWidthInVoxels;
    uint virtualBrickID = virtualBrickCoords.x + gParamBlock.virtualBricksPerAxis * (virtualBrickCoords.y + gParamBlock.virtualBricksPerAxis * virtualBrickCoords.z);
    gParamBlock.brickValidity.InterlockedOr(virtualBrickID * 4, voxelValid);
#endif
}
