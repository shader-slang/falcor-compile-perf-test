/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#include "Utils/NVAPI.slangh"

import DiffRendering.SharedTypes;

struct GradientsAggregator
{
    uint gradDim;
    uint hashSize;

    ByteAddressBuffer tmpGrads;
    RWByteAddressBuffer grads;

    void aggregateDirect(uint2 threadID)
    {
        if (threadID.x >= gradDim || threadID.y >= hashSize)
            return;

        uint index = threadID.y * gradDim + threadID.x;
        float value = asfloat(tmpGrads.Load(index * 4));
        grads.Store(threadID.x * 4, asuint(value));
    }

    void aggregateHashGrid(uint2 threadID)
    {
        if (threadID.x >= gradDim || threadID.y >= hashSize)
            return;

        uint index = threadID.y * gradDim + threadID.x;
        float value = asfloat(tmpGrads.Load(index * 4));
        grads.InterlockedAddF32(threadID.x * 4, value);
    }
}

ParameterBlock<GradientsAggregator> gAggregator;

[numthreads(256, 1, 1)]
void mainDirect(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    gAggregator.aggregateDirect(dispatchThreadID.xy);
}

[numthreads(64, 8, 1)]
void mainHashGrid(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    gAggregator.aggregateHashGrid(dispatchThreadID.xy);
}
