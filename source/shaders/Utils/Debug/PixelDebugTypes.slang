/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

///  Define the basic types that print() supports.
enum class PrintValueType
{
    Bool = 0,
    Int,
    Uint,
    Float,
};

struct PrintRecord
{
    uint msgHash; ///< String hash of print message.
    uint type;    ///< Value type (see PrintValueType).
    uint count;   ///< Number of components (1-4).
    uint _pad0;   ///< Padding.
    uint4 data;   ///< The data bits. The encoding is determined by the data type.
};

struct AssertRecord
{
    uint3 launchIndex; ///< Launch index for the assert.
    uint msgHash;      ///< String hash of assert message.
};

END_NAMESPACE_FALCOR
