/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import Utils.Attributes;
import ParamBlockDefinition;

// Global scope root and non-root resources
RWStructuredBuffer<float> result;
Buffer<uint> globalBufA;

#if USE_UAV
[root]
RWByteAddressBuffer globalTestBuffer; // Buffer bound to root descriptor (UAV)
#else
[root]
ByteAddressBuffer globalTestBuffer; // Buffer bound to root descriptor (SRV)
#endif

// Parameter block containing both root and non-root resources
ParameterBlock<TestBlock> gParamBlock;

[numthreads(256, 1, 1)]
void main(uint3 threadId: SV_DispatchThreadID)
{
    const uint i = threadId.x;

    // Add data read from the different buffers in the parameter block.
    // The values are multiplied by different constants to ensure that aliasing of the buffers to the wrong variables doesn't pass.
    float r = 0.f;
    r += gParamBlock.c0;
    r += gParamBlock.bufA[0].Load(i * 4);
    r += gParamBlock.bufA[1].Load(i * 4) * 2;
    r += gParamBlock.bufB[0][i] * 3;
    r += gParamBlock.bufB[1][i] * 4;
    r += gParamBlock.bufB[2][i] * 5;
    r += gParamBlock.bufC[0][i] * 6;
    r += gParamBlock.bufC[1][i] * 7;
    r += gParamBlock.bufC[2][i] * 8;
    r += gParamBlock.bufC[3][i] * 9;
    r += gParamBlock.testBuffer.Load(i * 4) * 10;

    // Add data read from buffers bound at the global scope.
    r += globalBufA[i] * 11;
    r += globalTestBuffer.Load(i * 4) * 12;

    result[i] = r;
}
