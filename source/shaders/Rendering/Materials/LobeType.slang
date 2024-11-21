/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

/** Flags representing the various lobes of a BSDF.
*/
// TODO: Specify the backing type when Slang issue has been resolved
// See https://gitlab-master.nvidia.com/slang/slang/-/issues/101
enum class LobeType // : uint32_t
{
    None                    = 0x00,

    DiffuseReflection       = 0x01,
    SpecularReflection      = 0x02,
    DeltaReflection         = 0x04,

    DiffuseTransmission     = 0x10,
    SpecularTransmission    = 0x20,
    DeltaTransmission       = 0x40,

    Diffuse                 = 0x11,
    Specular                = 0x22,
    Delta                   = 0x44,
    NonDelta                = 0x33,

    Reflection              = 0x0f,
    Transmission            = 0xf0,

    NonDeltaReflection      = 0x03,
    NonDeltaTransmission    = 0x30,

    All                     = 0xff,
};

END_NAMESPACE_FALCOR
