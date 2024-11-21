/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
RWStructuredBuffer<uint> result;

[numthreads(32, 32, 1)]
void main(uint3 threadId: SV_DispatchThreadID)
{
    uint idx = threadId.x % 1024;

    uint count = 0xffffffff;

    while (count-- > 0)
    {
        while (result[idx] != 0)
        {
            result[idx] += 1;
        }
        result[idx] = 1;
    }
}
