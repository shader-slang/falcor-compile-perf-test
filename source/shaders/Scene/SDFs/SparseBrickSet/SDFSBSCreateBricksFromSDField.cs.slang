/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"

import Utils.Math.AABB;
import Utils.Math.FormatConversion;
import Scene.SDFs.SDFVoxelCommon;
import Scene.SDFs.SparseBrickSet.BC4Encode;

static const uint kCompressionWidth = 4;
static const float kRootThree = sqrt(3.0f);

struct ParamBlock
{
    uint virtualGridWidth;
    uint virtualBrickCount;
    uint virtualBricksPerAxis;
    uint brickCount;
    uint brickWidthInVoxels;
    uint2 bricksPerAxis;
    Texture3D<float> sdfGrid;
    Buffer<uint> indirectionBuffer;

    RWStructuredBuffer<AABB> brickAABBs;
#if COMPRESS_BRICKS
    RWTexture2D<uint2> bricks;
#else
    RWTexture2D<float> bricks;
#endif
};

ParameterBlock<ParamBlock> gParamBlock;

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID)
{
    if (dispatchThreadID.x >= gParamBlock.virtualBrickCount) return;

    const uint virtualBrickID = dispatchThreadID.x;

    uint brickID = gParamBlock.indirectionBuffer[virtualBrickID];
    if (brickID == UINT32_MAX) return;

    // Calculate virtual brick coords.
    uint3 virtualBrickCoords = uint3(virtualBrickID, gParamBlock.virtualBricksPerAxis * gParamBlock.virtualBricksPerAxis, 0);
    virtualBrickCoords.z = virtualBrickCoords.x / virtualBrickCoords.y;  // = virtualBrickID / (virtualBricksPerAxis * virtualBricksPerAxis)
    virtualBrickCoords.x -= virtualBrickCoords.z * virtualBrickCoords.y; // = brickCoords.z * (virtualBricksPerAxis * virtualBricksPerAxis)
    virtualBrickCoords.y = virtualBrickCoords.x / gParamBlock.virtualBricksPerAxis;
    virtualBrickCoords.x = virtualBrickCoords.x % gParamBlock.virtualBricksPerAxis;

    // Calculate the min corner of the brick in the grid.
    uint brickWidthInValues = gParamBlock.brickWidthInVoxels + 1;
    uint3 brickGridCoords = virtualBrickCoords * gParamBlock.brickWidthInVoxels;

    // Calculate the AABB min and max corners for the brick.
    const float oneOverGridWidth = 1.0f / float(gParamBlock.virtualGridWidth);
    float3 brickAABBMin = -0.5f + float3(virtualBrickCoords * gParamBlock.brickWidthInVoxels) * oneOverGridWidth;
    float3 brickAABBMax = min(brickAABBMin + float(gParamBlock.brickWidthInVoxels) * oneOverGridWidth, 0.5f);

    gParamBlock.brickAABBs[brickID] = AABB(brickAABBMin, brickAABBMax);

    // Calculate the min corner of the brick in the brick texture.
    uint2 brickTextureCoords = uint2(brickID % gParamBlock.bricksPerAxis.x, brickID / gParamBlock.bricksPerAxis.x) * uint2(brickWidthInValues * brickWidthInValues, brickWidthInValues);

    // Write brick values.
    for (uint z = 0; z < brickWidthInValues; ++z)
    {
        for (uint y = 0; y < brickWidthInValues; y += kCompressionWidth)
        {
            for (uint x = 0; x < brickWidthInValues; x += kCompressionWidth)
            {
                uint2 voxelTextureCoords = brickTextureCoords + uint2(x + z * brickWidthInValues, y);
                uint2 blockTextureCoords = voxelTextureCoords / kCompressionWidth;

                uint3 blockGridCoords = brickGridCoords + uint3(x, y, z);

                int4x4 block;
                for (uint bY = 0; bY < kCompressionWidth && bY + y < brickWidthInValues; ++bY)
                {
                    for (uint bX = 0; bX < kCompressionWidth && bX + x < brickWidthInValues; ++bX)
                    {
                        uint3 voxelGridCoords = blockGridCoords + uint3(bX, bY, 0);

#if COMPRESS_BRICKS
                        if (all(voxelGridCoords < gParamBlock.virtualGridWidth))
                        {
                            float cornerDistance = gParamBlock.sdfGrid[voxelGridCoords];

                            // Convert to snorm.
                            float intScale = cornerDistance * 127.0f;
                            block[bY][bX] = int(intScale >= 0.0f ? intScale + 0.5f : intScale - 0.5f);
                        }
                        else
                        {
                            block[bY][bX] = 127;
                        }
#else
                        if (all(voxelGridCoords < gParamBlock.virtualGridWidth))
                        {
                            gParamBlock.bricks[voxelTextureCoords + uint2(bX, bY)] = gParamBlock.sdfGrid[voxelGridCoords];
                        }
                        else
                        {
                            gParamBlock.bricks[voxelTextureCoords + uint2(bX, bY)] = 1.0f;
                        }
#endif
                    }
                }

#if COMPRESS_BRICKS
                gParamBlock.bricks[blockTextureCoords] = compressBlock(block);
#endif
            }
        }
    }
}
