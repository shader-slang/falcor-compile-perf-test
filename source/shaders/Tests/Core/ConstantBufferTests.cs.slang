/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
RWStructuredBuffer<float> result;

struct Params
{
    int a;
    uint b;

    // This float3 field is not used in the test but it is here
    // to make sure we are currently using `scalar` layout on Vulkan.
    // Otherwise GLSL's default std140 rules will result in a different layout
    // from D3D12 and cause the test to fail.
    float3 val;

    float c;
};

cbuffer CB
{
    Params params1;
}

ConstantBuffer<Params> params2;

[numthreads(1, 1, 1)]
void testCbuffer1()
{
    result[0] = params1.a;
    result[1] = params1.b;
    result[2] = params1.c;
}

[numthreads(1, 1, 1)]
void testCbuffer2()
{
    result[0] = params2.a;
    result[1] = params2.b;
    result[2] = params2.c;
}
