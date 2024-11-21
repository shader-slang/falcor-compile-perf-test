/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#pragma once
#include "Utils/HostDeviceShared.slangh"

/** This tests uses Slang's 'reinterpret' call to cast a fixed-size struct
    between different struct types containing a mix of 16/32-bit basic types.
    Note all 4B types are 4B aligned to avoid issues due to packing rules.
*/

BEGIN_NAMESPACE_FALCOR

struct Blob
{
    uint4 data[3];
};

struct A
{
    uint a;
    float b;
    float16_t c;
    int16_t d;
    int2 e;
    float3 f;
    uint g;
    float16_t3 h;
    uint16_t i;
    float16_t2 j;
};

struct B
{
    float a[12];
};

struct C
{
    uint4 a;
    float16_t4 b;
    float d[4];
    float16_t4 c;
};

struct D
{
    int16_t a;
    int16_t b;
    uint3 c;
    float d;
    float4 e;
    float3 r;
};

struct E
{
    float16_t4 a[6];
};

struct F
{
    int a;
    float16_t4 b[5];
    uint16_t c;
    float16_t d;
};

#ifndef HOST_CODE
StructuredBuffer<Blob> data;

RWStructuredBuffer<A> resultA;
RWStructuredBuffer<B> resultB;
RWStructuredBuffer<C> resultC;
RWStructuredBuffer<D> resultD;
RWStructuredBuffer<E> resultE;
RWStructuredBuffer<F> resultF;

[numthreads(32, 1, 1)]
void main(uint3 threadID: SV_DispatchThreadID)
{
    const uint i = threadID.x;

    Blob blob = data[i];

    B b = reinterpret<B, Blob>(blob);
    C c = reinterpret<C, B>(b);
    D d = reinterpret<D, C>(c);
    E e = reinterpret<E, D>(d);
    F f = reinterpret<F, E>(e);
    A a = reinterpret<A, F>(f);

    resultA[i] = a;
    resultB[i] = b;
    resultC[i] = c;
    resultD[i] = d;
    resultE[i] = e;
    resultF[i] = f;
}
#endif

END_NAMESPACE_FALCOR
