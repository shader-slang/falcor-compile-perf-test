/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/

/** Enums for texture level-of-detail -- see TexLODHelpers.slang
*/

#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

/** This enum is shared between CPU/GPU.
    It enumerates the different the different texture LOD modes.
*/
enum class TexLODMode : uint32_t
{
    Mip0 = 0,
    RayCones = 1,
    RayDiffs = 2,
    Stochastic = 3,
};

FALCOR_ENUM_INFO(TexLODMode, {
    { TexLODMode::Mip0, "Mip0" },
    { TexLODMode::RayCones, "RayCones" },
    { TexLODMode::RayDiffs, "RayDiffs" },
    { TexLODMode::Stochastic, "Stochastic" },
});
FALCOR_ENUM_REGISTER(TexLODMode);

/** This enum is shared between CPU/GPU.
    Both Combo and Unified avoid computing surface spread from the G-buffer.
    Combo uses ray differentials to compute the spread at the first hit and then average edge curvature for all hits after that.
    Unified uses interpolated edge curvatures for the first hit and then average edge curvatures for all hits after that.
*/
enum class RayConeMode : uint32_t
{
    Combo = 0,
    Unified = 1,
};

FALCOR_ENUM_INFO(RayConeMode, {
    { RayConeMode::Combo, "Combo" },
    { RayConeMode::Unified, "Unified" },
});
FALCOR_ENUM_REGISTER(RayConeMode);

END_NAMESPACE_FALCOR
