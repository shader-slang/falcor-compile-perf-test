/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
import Internal.Scene2.SharedTypes;
import Utils.Math.MathHelpers;

#include "Internal/Scene2/Scene2IDs.slangh"

StructuredBuffer<float4x4> gSceneGraphXformWorldFromInstance;
StructuredBuffer<SceneGraphNodeID> gSrcIndex;
RWStructuredBuffer<float4x4> gSceneXformWorldFromInstance;
RWStructuredBuffer<float4x4> gSceneXformInstanceFromWorld;
RWStructuredBuffer<SlangRtInstanceDesc> gRtInstanceDesc;

cbuffer CB
{
    uint dstCount;
}

/**
 * @brief Used to copy matrices from SceneGraph buffer, to RtInstanceDesc buffer.
 */
[shader("compute")]
[numthreads(1024, 1, 1)]
void copyMatrices(uint3 threadIdx: SV_DispatchThreadID)
{
    const uint dstIndex = threadIdx.x;
    if (dstIndex >= dstCount)
        return;
    const SceneGraphNodeID srcIndex = gSrcIndex[dstIndex];
    float4x4 m = gSceneGraphXformWorldFromInstance[srcIndex.get()];
    gSceneXformWorldFromInstance[dstIndex] = m;

    float4x4 inv = inverse(m);
    gSceneXformInstanceFromWorld[dstIndex] = inv;

    gRtInstanceDesc[dstIndex].transform = float3x4(m);
}
