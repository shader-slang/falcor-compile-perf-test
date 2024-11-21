/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import Tests.Slang.ShaderStringImported;

RWStructuredBuffer<uint> result;

[numthreads(32, 1, 1)]
void main(uint3 threadID: SV_DispatchThreadID)
{
    // The function test() is defined in the imported module,
    // which in turn calls a generated function f().
    uint y = test(threadID.x);

    result[threadID.x] = y;
}
