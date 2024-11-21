/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#pragma once
#include "Utils/HostDeviceShared.slangh"

#ifdef HOST_CODE
#include "Utils/Math/AABB.h"
#else
import Utils.Math.AABB;
#endif

BEGIN_NAMESPACE_FALCOR

struct SDFSVSVoxel
{
    uint4 packedValuesSlices[4];    ///< 4x4 value slices in the x-direction of the voxel, packed as 8-bit snorms.
    uint validNeighborsMask;        ///< Low 27 bits used as validity mask for neighbors, wastes one bit of the mask, but fine for now.
};

struct SDFSVOHashTableVoxel
{
    uint2 locationCode;     ///< The voxel location code.
    uint svoOffset;         ///< The offset of this voxel in the SVO.
    uint validMask;         ///< The valid mask of the voxel, encoding what children are valid, padded so voxel is multiple of 4 bytes in size.
    uint2 packedValues;     ///< The eight corner values of the SDFVoxel, packed as 8-bit snorms.
};

struct SDFSVOVoxel
{
    uint relationData;      ///< Relation data for the voxel encoding the index its first child in the SVO and the validity of its children.
    uint2 locationCode;     ///< The location code of the voxel.
    uint2 packedValues;     ///< The eight corner values of the SDFVoxel, packed as 8-bit snorms.
};

END_NAMESPACE_FALCOR
