/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
RWStructuredBuffer<float> result;

struct S
{
    float a;
};

ParameterBlock<S> gParamBlock;

[numthreads(1, 1, 1)]
void main()
{
    result[0] = gParamBlock.a;
}
