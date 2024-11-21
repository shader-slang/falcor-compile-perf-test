/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import Scene.SDFs.SparseVoxelOctree.SDFSVOHashTable;

cbuffer CB
{
    uint gLocationCodeStartOffset;
    uint gVoxelCount;
};

ParameterBlock<SDFSVOHashTable> gVoxelHashTable;
RWByteAddressBuffer gLocationCodes;

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x > gVoxelCount) return;

    uint2 locationCode = gLocationCodes.Load2(8 * (gLocationCodeStartOffset + dispatchThreadID.x));
    gVoxelHashTable.setSVOOffset(locationCode, dispatchThreadID.x);
}
