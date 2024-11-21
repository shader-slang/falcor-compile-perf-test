/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
RWStructuredBuffer<uint> result;
RWTexture3D<uint> tex3D_uav;
Texture3D<uint> tex3D_srv;

[numthreads(16, 16, 1)]
void testTexture3DWrite(uint3 threadId: SV_DispatchThreadID)
{
    tex3D_uav[threadId] = threadId.x * threadId.y * threadId.z + 577;
}

[numthreads(16, 16, 1)]
void testTexture3DRead(uint3 threadId: SV_DispatchThreadID)
{
    uint index = threadId.z * 256 + threadId.y * 16 + threadId.x;
    result[index] = tex3D_srv[threadId];
}
