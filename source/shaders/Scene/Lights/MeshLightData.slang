/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

/** Describes a mesh area light.

    Note that due to mesh merging etc. it's not guaranteed that a mesh light
    represents a single localized light source in the scene.
    We should not make any assumptions of the extents or geometry of a mesh light.
*/
struct MeshLightData
{
    uint        instanceID          = kInvalidIndex;    ///< Geometry instance ID in the scene.
    uint        triangleOffset      = kInvalidIndex;    ///< Offset into LightCollection's global list of emissive triangles.
    uint        triangleCount       = 0;                ///< Number of triangles in mesh light.
    uint        materialID          = kInvalidIndex;    ///< Material ID (index into scene.materials array).

    static constexpr uint kInvalidIndex = 0xffffffff;
};

END_NAMESPACE_FALCOR
