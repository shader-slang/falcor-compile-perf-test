/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
RWStructuredBuffer<uint4> result;

Texture2D<float4> texUnorm;
Texture2D<uint4> texUnormAsUint;
Texture2D<uint4> texUint;

[numthreads(256, 1, 1)]
void testLoadFormat(uint3 threadId: SV_DispatchThreadID)
{
    float4 f = texUnorm[threadId.xy];
    uint4 u = texUnormAsUint[threadId.xy];
    uint4 v = texUint[threadId.xy];

    result[threadId.x] = { asuint(f.x), u.x, (uint)(f.x * 255.f), v.x };
}

[numthreads(1, 1, 1)]
void testLoadMips(uint3 threadId: SV_DispatchThreadID)
{
    result[0] = (uint4)(texUnorm.Load(int3(0, 0, 0)) * 255.f);
    result[1] = (uint4)(texUnorm.Load(int3(0, 0, 1)) * 255.f);
    result[2] = (uint4)(texUnorm.Load(int3(0, 0, 2)) * 255.f);
}
