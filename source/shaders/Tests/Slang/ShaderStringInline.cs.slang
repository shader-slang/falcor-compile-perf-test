/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
RWStructuredBuffer<uint> result;

struct Test
{
    // The definition of A is generated into the same translation unit.
    // The definition should be visible here and we can use the type directly.
    A moduleA;

    void execute(uint threadID) { result[threadID] = moduleA.f(threadID); }
};

ParameterBlock<Test> gTest;

[numthreads(32, 1, 1)]
void main(uint3 threadID: SV_DispatchThreadID)
{
    gTest.execute(threadID.x);
}
