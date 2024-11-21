/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

enum class ColorMap : uint32_t
{
    Grey,
    Jet,
    Viridis,
    Plasma,
    Magma,
    Inferno,
};

FALCOR_ENUM_INFO(
    ColorMap,
    {
        { ColorMap::Grey, "Grey" },
        { ColorMap::Jet, "Jet" },
        { ColorMap::Viridis, "Viridis" },
        { ColorMap::Plasma, "Plasma" },
        { ColorMap::Magma, "Magma" },
        { ColorMap::Inferno, "Inferno" },
    }
);
FALCOR_ENUM_REGISTER(ColorMap);

struct ColorMapParams
{
    float minValue = 0.f;
    float maxValue = 1.f;
};

END_NAMESPACE_FALCOR
