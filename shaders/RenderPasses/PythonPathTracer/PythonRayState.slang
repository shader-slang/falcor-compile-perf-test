/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
__exported import Plugins.MaterialDataGenerator.MaterialDataShared;

static const uint kUnmappedIndex = 0xffffffff;

struct PythonRayState
{
    RWStructuredBuffer<MaterialSample> samples;
    RWBuffer<uint> compactionIdx;

    uint elemCount;
    uint compactedCount;

    void markActive(uint idx) { compactionIdx[idx] = 1; }
    uint mapIdx(uint idx)
    {
        uint a = compactionIdx[idx];
        uint b = (idx + 1 == elemCount) ? compactedCount : compactionIdx[idx + 1];
        return a == b ? kUnmappedIndex : a;
    }
};
