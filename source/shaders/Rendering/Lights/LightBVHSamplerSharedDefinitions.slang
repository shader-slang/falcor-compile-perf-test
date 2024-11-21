/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

enum class SolidAngleBoundMethod : uint32_t
{
    BoxToCenter = 1,
    BoxToAverage,
    Sphere,
};

FALCOR_ENUM_INFO(SolidAngleBoundMethod, {
    { SolidAngleBoundMethod::BoxToCenter, "BoxToCenter" },
    { SolidAngleBoundMethod::BoxToAverage, "BoxToAverage" },
    { SolidAngleBoundMethod::Sphere, "Sphere" },
});
FALCOR_ENUM_REGISTER(SolidAngleBoundMethod);

END_NAMESPACE_FALCOR
