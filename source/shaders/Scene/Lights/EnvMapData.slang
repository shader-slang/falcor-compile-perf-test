/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

/** This is a host/device structure that describes the environment map proeprties.
*/
struct EnvMapData
{
    float3x4    transform;              ///< Local to world transform.
    float3x4    invTransform;           ///< World to local transform.

    float3      tint = {1.f, 1.f, 1.f}; ///< Color tint
    float       intensity = 1.f;        ///< Radiance scale
};

END_NAMESPACE_FALCOR
