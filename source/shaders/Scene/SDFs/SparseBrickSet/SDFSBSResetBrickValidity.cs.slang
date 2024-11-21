/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"

struct ParamBlock
{
    uint virtualBrickCount;
    Buffer<uint> brickValidity;
    RWBuffer<uint> indirectionBuffer;
};

ParameterBlock<ParamBlock> gParamBlock;

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID)
{
    if (dispatchThreadID.x >= gParamBlock.virtualBrickCount)
        return;

    uint virtualBrickID = dispatchThreadID.x;
    uint valid = gParamBlock.brickValidity[virtualBrickID];

    if (valid == 0)
    {
        gParamBlock.indirectionBuffer[virtualBrickID] = UINT32_MAX;
    }
}
