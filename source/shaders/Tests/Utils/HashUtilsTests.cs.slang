/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import Utils.Math.HashUtils;

RWBuffer<uint> result;

[numthreads(16, 16, 1)]
void testJenkinsHash_PerfectHash(uint3 threadId: SV_DispatchThreadID)
{
    // Test that hash function is a perfect hash (no collisions).
    // We run 2^32 threads, and record the output as 1 bit per possible value (2^27 dwords).
    uint i = (threadId.y << 16) | threadId.x; // i in [0, 2^32)
    uint h = jenkinsHash(i);

    uint k = h >> 5;
    uint b = h & 0x1f;
    InterlockedOr(result[k], 1u << b);
}

[numthreads(16, 16, 1)]
void testJenkinsHash(uint3 threadId: SV_DispatchThreadID)
{
    uint i = (threadId.y << 16) | threadId.x;
    result[i] = jenkinsHash(i);
}
