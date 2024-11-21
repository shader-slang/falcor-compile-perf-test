/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import Scene.SDFs.SDFVoxelCommon;
import Scene.SDFs.SDFVoxelTypes;
import Scene.SDFs.SparseVoxelOctree.SDFSVOHashTable;

cbuffer CB
{
    uint gLocationCodeStartOffset;
    uint gVoxelCount;
};

ParameterBlock<SDFSVOHashTable> gVoxelHashTable;
RWByteAddressBuffer gLocationCodes;

RWStructuredBuffer<SDFSVOVoxel> gSVO;

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x > gVoxelCount) return;

    // Load the location code for the current voxel.
    const uint2 locationCode = gLocationCodes.Load2(8 * (gLocationCodeStartOffset + dispatchThreadID.x));

    uint svoOffset;
    uint validMask;
    uint2 packedValues;
    // Load the current voxel values, there must be one voxel per thread so we don't need to check whether if SDFSVOHashTable::lookup() returns true or false.
    if (gVoxelHashTable.lookup(locationCode, svoOffset, validMask, packedValues))
    {
        SDFSVOVoxel currentVoxel;
        currentVoxel.locationCode = locationCode;
        currentVoxel.relationData = validMask;
        currentVoxel.packedValues = packedValues;

        // Find the ID of the first valid child.
        for (uint firstValidChildID = 0; firstValidChildID < 8; firstValidChildID++)
        {
            if ((currentVoxel.relationData & (1 << firstValidChildID)) > 0)
            {
                // Create the location code for the child.
                uint2 childLocationCode;
                SDFVoxelCommon::createChildLocationCode(currentVoxel.locationCode, firstValidChildID, childLocationCode);
                childLocationCode.y |= (1 << 31);

                uint childSVOOffset;
                uint childValidMask;
                uint2 childPackedValues;

                // The valid mask guarantees that the child voxel exist.
                gVoxelHashTable.lookup(childLocationCode, childSVOOffset, childValidMask, childPackedValues);

                // Encode the relative child offset into the top 24 bits of relation data.
                currentVoxel.relationData |= (childSVOOffset << 8);
                break;
            }
        }

        gSVO[svoOffset] = currentVoxel;
    }
}
