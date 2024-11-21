/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import Utils.Attributes;

RWStructuredBuffer<uint> result;

struct S
{
    uint4 v;
};

#if USE_ROOT_DESC
[root]
ByteAddressBuffer buffer;
[root]
StructuredBuffer<S> structuredBuffer;
[root]
StructuredBuffer<uint> structuredBufferUint;
#else
ByteAddressBuffer buffer;
StructuredBuffer<S> structuredBuffer;
StructuredBuffer<uint> structuredBufferUint;
#endif

cbuffer CB
{
    uint elemCount;
}

[numthreads(256, 1, 1)]
void testReadRaw(uint3 threadId: SV_DispatchThreadID)
{
    // Copy the last 256 elements of the buffer.
    uint i = threadId.x;
    uint addr = (elemCount - 256 + i) * 4;
    result[i] = buffer.Load(addr);
}

[numthreads(256, 1, 1)]
void testReadStructured(uint3 threadId: SV_DispatchThreadID)
{
    // Copy the last 256 elements of the buffer.
    uint i = threadId.x;
    uint index = (elemCount - 256 + i);
    result[i] = structuredBuffer[index].v.x;
}

[numthreads(256, 1, 1)]
void testReadStructuredUint(uint3 threadId: SV_DispatchThreadID)
{
    // Copy the last 256 elements of the buffer.
    uint i = threadId.x;
    uint index = (elemCount - 256 + i);
    result[i] = structuredBufferUint[index];
}
