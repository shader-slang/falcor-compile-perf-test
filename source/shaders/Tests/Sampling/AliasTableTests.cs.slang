/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import Utils.Sampling.AliasTable;

StructuredBuffer<float> random;

RWStructuredBuffer<uint> sampleResult;
RWStructuredBuffer<float> weightResult;

cbuffer CB
{
    AliasTable aliasTable;
    uint resultCount;
};

[numthreads(256, 1, 1)]
void testAliasTableSample(uint3 threadId: SV_DispatchThreadID)
{
    const uint idx = threadId.x;
    if (idx >= resultCount)
        return;
    sampleResult[idx] = aliasTable.sample(float2(random[idx * 2], random[idx * 2 + 1]));
}

[numthreads(256, 1, 1)]
void testAliasTableWeight(uint3 threadId: SV_DispatchThreadID)
{
    const uint idx = threadId.x;
    if (idx >= resultCount)
        return;
    weightResult[idx] = aliasTable.getWeight(idx);
}
