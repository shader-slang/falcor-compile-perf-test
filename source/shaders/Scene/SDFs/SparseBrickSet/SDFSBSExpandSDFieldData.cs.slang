/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"

static const uint kGroupWidth = GROUP_WIDTH;

struct SDFieldData
{
    uint gridWidthInValues;
    uint offset;
    float expansionFactor;
    Texture3D<float> oldSDField;
    RWTexture3D<float> newSDField;
};

ParameterBlock<SDFieldData> gSDFieldData;

[numthreads(kGroupWidth, kGroupWidth, kGroupWidth)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (any(dispatchThreadID >= gSDFieldData.gridWidthInValues)) return;

    const uint3 valueCoords = dispatchThreadID + gSDFieldData.offset;
    gSDFieldData.newSDField[valueCoords] = gSDFieldData.oldSDField[dispatchThreadID] * gSDFieldData.expansionFactor;
}
