/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#ifdef IMPORT_FROM_MAIN
// Test importing another module from both here and from the generated code.
// There has been issues around duplicated imports.
import Tests.Slang.ShaderStringUtil;
#endif

RWStructuredBuffer<uint> result;

[numthreads(32, 1, 1)]
void main(uint3 threadID: SV_DispatchThreadID)
{
    // The code for f() is generated into the same translation unit.
    // The generated code calls a function imported from another module.
    result[threadID.x] = f(threadID.x);
}
