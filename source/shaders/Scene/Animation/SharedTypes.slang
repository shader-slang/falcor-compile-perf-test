/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

struct InterpolationInfo
{
    uint2 keyframeIndices;
    float t;
};

struct PerCurveMetadata
{
    uint indexCount;
    uint indexOffset;
    uint vertexCount;
    uint vertexOffset;
};

struct PerMeshMetadata
{
    // Input
    uint keyframeBufferOffset;

    // Output
    uint sceneVbOffset;
    uint prevVbOffset;
    uint vertexCount;
};

END_NAMESPACE_FALCOR
