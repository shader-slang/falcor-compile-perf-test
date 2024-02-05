/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
import Plugins.MaterialDataGenerator.MaterialDataShared;

cbuffer CB
{
    uint nSamples;
    RWStructuredBuffer<MaterialSample> sampleBuffer;
    StructuredBuffer<float2> uvBuffer;
    StructuredBuffer<float3> wiBuffer;
    StructuredBuffer<float3> woBuffer;
}

[numthreads(256, 1, 1)]
void copy(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    const uint idx = dispatchThreadId.x;
    if (idx >= nSamples)
        return;

    MaterialSample ms = sampleBuffer[idx];
    ms.uv = uvBuffer[idx];
    ms.wi = wiBuffer[idx];
    ms.wo = woBuffer[idx];
    sampleBuffer[idx] = ms;
}
