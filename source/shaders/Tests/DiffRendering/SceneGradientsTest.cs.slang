/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import DiffRendering.SceneGradients;

cbuffer CB
{
    uint2 sz;
    uint hashSize;
}

ByteAddressBuffer grads;
RWStructuredBuffer<float> result;

[numthreads(4, 16, 1)]
void atomicAdd(uint3 threadID: SV_DispatchThreadID)
{
    if (any(threadID.xy >= sz))
        return;
    gSceneGradients.atomicAddGrad(GradientType::Material, threadID.x, threadID.y % hashSize, pow(10.f, float(threadID.x)));
}

[numthreads(4, 1, 1)]
void testAggregateGradients(uint3 threadID: SV_DispatchThreadID)
{
    if (threadID.x >= sz.x)
        return;
    float value = asfloat(grads.Load(threadID.x * 4));
    result[threadID.x] = value;
}
