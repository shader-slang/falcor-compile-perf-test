/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
RWStructuredBuffer<float> result;

struct S
{
    float d[8];
};

// These descriptors alias the same raw buffer.
StructuredBuffer<float> bufA1;
StructuredBuffer<float4> bufA2;
StructuredBuffer<S> bufA3;

// These descriptors alias the same raw buffer.
RWStructuredBuffer<float> bufB1;
RWStructuredBuffer<float4> bufB2;
RWStructuredBuffer<S> bufB3;

// These descriptors alias the same structured buffer.
StructuredBuffer<uint> bufStruct1;
StructuredBuffer<uint4> bufStruct2;
StructuredBuffer<S> bufStruct3;

[numthreads(32, 1, 1)]
void testRead(uint3 threadId: SV_DispatchThreadID)
{
    uint i = threadId.x;
    result[i] = bufA1[i];
    result[i + 32] = bufA2[i / 4][i % 4];
    result[i + 64] = bufA3[i / 8].d[i % 8];
}

[numthreads(32, 1, 1)]
void testReadWrite(uint3 threadId: SV_DispatchThreadID)
{
    uint i = threadId.x;
    bufB1[i] = (float)(32 - i);
    bufB2[i / 4 + 8][i % 4] = (float)(32 - i);
    bufB3[i / 8 + 8].d[i % 8] = (float)(32 - i);
}

[numthreads(32, 1, 1)]
void testStructRead(uint3 threadId: SV_DispatchThreadID)
{
    uint i = threadId.x;
    result[i] = asfloat(bufStruct1[i]);
    result[i + 32] = asfloat(bufStruct2[i / 4][i % 4]);
    result[i + 64] = bufStruct3[i / 8].d[i % 8];
}
