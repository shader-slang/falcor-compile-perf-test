/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

struct A
{
    uint scalar;

    SETTER_DECL void setScalar(uint x) { scalar = x; }
    uint getScalar() { return scalar; }
};

struct B : A
{
    float3 vector;

    SETTER_DECL void setVector(float3 v) { vector = v; }
    float3 getVector() { return vector; }
};

#ifndef HOST_CODE
RWStructuredBuffer<B> result;
Buffer<uint> data;

[numthreads(1, 1, 1)]
void main()
{
    B test = {};

    uint s = data[0];
    float3 v = asfloat(uint3(data[1], data[2], data[3]));

    test.setScalar(s);
    test.setVector(v);

    B test2 = {};
    test2.setScalar(test.getScalar());
    test2.setVector(test.getVector());

    result[0] = test2;
}
#endif

END_NAMESPACE_FALCOR
