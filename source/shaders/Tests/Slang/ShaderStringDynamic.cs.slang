/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import Tests.Slang.ShaderStringDynamicType;

cbuffer CB
{
    uint type;
}

RWStructuredBuffer<uint> result;

[numthreads(32, 1, 1)]
void main(uint3 threadID: SV_DispatchThreadID)
{
    // Dynamically create an instance of the given type.
    // Note we don't need any data so passing in a dummy int.
    // The type is defined in ShaderStringDynamicType and internally calls into generated code.
    int dummyData = 0;
    let obj = createDynamicObject<IDynamicType, int>(int(type), dummyData);

    uint y = obj.eval(threadID.x);

    result[threadID.x] = y;
}
