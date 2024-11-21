/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/

/** Test basic compute shader support for the given shader model.
 */
RWStructuredBuffer<uint> result;

[numthreads(256, 1, 1)]
void main(uint3 threadID: SV_DispatchThreadID)
{
    uint i = threadID.x;
    result[i] = 3 * i;
}
