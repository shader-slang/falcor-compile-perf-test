/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

/** Test enum declarations on different forms, shared on CPU/GPU.

    Note that enums in Slang are currently always scoped, using the `.` operator,
    `enum Type { A, B };` is therefore the same as `enum class Type { A, B };`
    on the GPU side. We should use scoped enums by default as Slang is unlikely
    to support unscoped enums anytime soon.
*/

enum class Type1
{
    A,
    B,
    C,
    D,
};

enum class Type2
{
    A,
    B,
    C = 20,
    D,
};

enum class Type3 : uint32_t /* test typed enum */
{
    A = 0x01,
    B = 0x02,
    C = 0x04,
    D = 0x08,
};

END_NAMESPACE_FALCOR
