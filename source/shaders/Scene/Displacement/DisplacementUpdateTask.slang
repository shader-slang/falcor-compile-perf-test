/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

/** Represents a task to compute AABBs for a list of triangles.
*/
struct DisplacementUpdateTask
{
    static const uint kThreadCount = 16384; ///< Number of threads launched per task.

    uint meshID;            ///< Mesh ID to read vertices from.
    uint triangleIndex;     ///< First triangle to process.
    uint AABBIndex;         ///< First AABB to write.
    uint count;             ///< Number of triangles to process.
};

END_NAMESPACE_FALCOR
