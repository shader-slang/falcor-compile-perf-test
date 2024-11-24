/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

enum class ToneMapperOperator : uint32_t
{
    Linear,           ///< Linear mapping
    Reinhard,         ///< Reinhard operator
    ReinhardModified, ///< Reinhard operator with maximum white intensity
    HejiHableAlu,     ///< John Hable's ALU approximation of Jim Heji's filmic operator
    HableUc2,         ///< John Hable's filmic tone-mapping used in Uncharted 2
    Aces,             ///< Aces Filmic Tone-Mapping
};

FALCOR_ENUM_INFO(
    ToneMapperOperator,
    {
        { ToneMapperOperator::Linear, "Linear" },
        { ToneMapperOperator::Reinhard, "Reinhard" },
        { ToneMapperOperator::ReinhardModified, "ReinhardModified" },
        { ToneMapperOperator::HejiHableAlu, "HejiHableAlu" },
        { ToneMapperOperator::HableUc2, "HableUc2" },
        { ToneMapperOperator::Aces, "Aces" },
    }
);
FALCOR_ENUM_REGISTER(ToneMapperOperator);

/**
 * Tone mapper parameters shared between host and device.
 * Make sure struct layout follows the HLSL packing rules as it is uploaded as a memory blob.
 * Do not use bool's as they are 1 byte in Visual Studio, 4 bytes in HLSL.
 * https://msdn.microsoft.com/en-us/library/windows/desktop/bb509632(v=vs.85).aspx
 */
struct ToneMapperParams
{
    float whiteScale;
    float whiteMaxLuminance;
    float _pad0;
    float _pad1;
    float3x4 colorTransform;
};

END_NAMESPACE_FALCOR
