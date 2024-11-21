/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import Utils.Attributes;

/** Unit tests for testing a buffer bound as root descriptor inside a struct,
    where the struct is placed in a constant buffer.
*/

RWStructuredBuffer<uint> result;

struct T
{
    uint value;
};

struct S
{
    Buffer<uint> buf;
    RWBuffer<uint> rwBuf;
#if USE_UAV
    [root]
    RWStructuredBuffer<T> rootBuf; // Buffer bound to root descriptor (UAV)
#else
    [root]
    StructuredBuffer<T> rootBuf; // Buffer bound to root descriptor (SRV)
#endif
};

cbuffer CB
{
    S data;
}

[numthreads(256, 1, 1)]
void main(uint3 threadId: SV_DispatchThreadID)
{
    const uint i = threadId.x;

    // Add data read from the different buffers together.
    // The values are multiplied by different constants to ensure that aliasing of the buffers to the wrong variables doesn't pass.
    uint r = 0;
    r += data.buf[i];
    r += data.rwBuf[i] * 2;
    r += data.rootBuf[i].value * 3;

    result[i] = r;
}
