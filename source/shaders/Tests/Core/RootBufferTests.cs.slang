/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import Utils.Attributes;

/** Unit tests for binding buffers using root descriptors.
 */

cbuffer CB
{
    uint c0;
    float c1;
}

struct S
{
    float a;
    uint b;
};

// Place the root buffer in the middle of some other buffers to ensure binding the root buffer
// doesn't intefere with the bindings before or after.
RWStructuredBuffer<float> result;
ByteAddressBuffer rawBuffer;
StructuredBuffer<S> structBuffer;

#if USE_UAV
[root]
RWByteAddressBuffer testBuffer; // Buffer bound to root descriptor (UAV)
#else
[root]
ByteAddressBuffer testBuffer; // Buffer bound to root descriptor (SRV)
#endif

RWBuffer<uint> typedBufferUint;
Buffer<float4> typedBufferFloat4;

[numthreads(256, 1, 1)]
void main(uint3 threadId: SV_DispatchThreadID)
{
    const uint i = threadId.x;

    // Add data read from the different buffers together.
    // The values are multiplied by different constants to ensure that aliasing of the buffers to the wrong variables doesn't pass.
    float r = 0.f;
    r += c0;
    r += c1;
    r += rawBuffer.Load(i * 4);
    r += typedBufferUint[i] * 2;
    r += typedBufferFloat4[i].z * 3;
    r += structBuffer[i].a * 4;
    r += structBuffer[i].b * 5;
    r += testBuffer.Load(i * 4) * 6;

    result[i] = r;
}
