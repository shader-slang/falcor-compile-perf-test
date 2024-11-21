/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import Utils.Sampling.LowDiscrepancy.HammersleySequence;

RWStructuredBuffer<float> result;

cbuffer TestCB
{
    int resultSize;
};

[numthreads(1, 1, 1)]
void testRadicalInverse()
{
    for (int i = 0; i < resultSize; ++i)
    {
        result[i] = radicalInverse(i);
    }
}
