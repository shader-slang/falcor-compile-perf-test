/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
StructuredBuffer<float> bufferSrv;
RWStructuredBuffer<float> bufferUav;
RWByteAddressBuffer counter;

cbuffer CB
{
    uint3 dim;
    uint offset;
}

uint getIndex(uint3 threadID)
{
    return (threadID.x * dim.y + threadID.y) * dim.z + threadID.z;
}

[numthreads(16, 16, 1)]
void writeBuffer(uint3 threadID: SV_DispatchThreadID)
{
    if (any(threadID >= dim))
        return;

    uint idx = getIndex(threadID);
    bufferUav[idx] = (float)(offset + idx);
}

[numthreads(1, 16, 16)]
void readBuffer(uint3 threadID: SV_DispatchThreadID)
{
    if (any(threadID >= dim))
        return;

    uint idx = getIndex(threadID);
    float expectedValue = (float)(offset + idx);
    float value = bufferSrv[idx];

    if (value == expectedValue)
        counter.InterlockedAdd(0, 1);
}
