/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

enum class HitType : uint32_t
{
    None                = 0,    ///< No hit.
    Triangle            = 1,    ///< Triangle hit.
    Volume              = 2,    ///< Volume hit.

    // The following hit types are only available if hit info compression is disabled.

    DisplacedTriangle   = 3,    ///< Displaced triangle hit.
    Curve               = 4,    ///< Curve hit.
    SDFGrid             = 6,    ///< SDF grid hit.

    //
    // Add new hit types here
    //

    Count // Must be last
};

END_NAMESPACE_FALCOR
