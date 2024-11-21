/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

/**
 * BSDFOptimizer parameters shared between host and device.
 * Make sure struct layout follows the HLSL packing rules as it is uploaded as a memory blob.
 * Do not use bool's as they are 1 byte in Visual Studio, 4 bytes in HLSL.
 * https://msdn.microsoft.com/en-us/library/windows/desktop/bb509632(v=vs.85).aspx
 */
struct BSDFOptimizerParams
{
    uint2 frameDim = { 0, 0 };
    uint frameCount = 0;
    int _pad0;

    float2 initViewPortOffset;
    float2 diffViewPortOffset;

    float2 refViewPortOffset;
    float2 viewPortScale;

    uint2 bsdfTableDim = { 0, 0 };
    uint initMaterialID = 0;
    uint refMaterialID = 1;
};

END_NAMESPACE_FALCOR
