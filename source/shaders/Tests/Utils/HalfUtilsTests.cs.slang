/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import Utils.Math.HalfUtils;

cbuffer CB
{
    uint testSize;
}

StructuredBuffer<float> inputFloat;
StructuredBuffer<uint> inputUint;

RWStructuredBuffer<uint> resultUint;
RWStructuredBuffer<float> resultFloat;

/** Converts a sequence of fp32 to fp16 using default rounding.
 */
[numthreads(256, 1, 1)]
void testFP32ToFP16(uint3 threadId: SV_DispatchThreadID)
{
    uint i = threadId.x;
    if (i >= testSize)
        return;

    resultUint[i] = f32tof16(inputFloat[i]);
}

/** Converts a sequence of fp16 numbers to fp32.
 */
[numthreads(256, 1, 1)]
void testFP16ToFP32(uint3 threadId: SV_DispatchThreadID)
{
    uint i = threadId.x;
    if (i >= testSize)
        return;

    resultFloat[i] = f16tof32(inputUint[i]);
}

/** Converts a sequence of fp32 numbers to fp16 using conservative rounding up/down.
 */
[numthreads(256, 1, 1)]
void testFP32ToFP16ConservativeRounding(uint3 threadId: SV_DispatchThreadID)
{
    uint i = threadId.x;
    if (i >= testSize)
        return;

    float v = inputFloat[i];
    resultUint[2 * i + 0] = f32tof16_roundUp(v);
    resultUint[2 * i + 1] = f32tof16_roundDown(v);
}

#if 0
// TODO: Currently disabled until we figure out the rounding modes and have a matching CPU library.
[numthreads(256, 1, 1)]
void testFP16RoundingMode(uint3 threadId: SV_DispatchThreadID)
{
    uint i = threadId.x;
    if (i >= testSize)
        return;

    resultFloat[i] = f16tof32(f32tof16(inputFloat[i])); // TODO: Use precise keyword when available.
}
#endif
