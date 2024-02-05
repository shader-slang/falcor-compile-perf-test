/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
import RenderPasses.PythonPathTracer.PythonRayState;

cbuffer CB
{
    PythonRayState gSrc;
    PythonRayState gDst;
}

[numthreads(256, 1, 1)]
void compact(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    const uint idx = dispatchThreadId.x;
    if (idx >= gSrc.elemCount)
        return;

    uint a = gSrc.mapIdx(idx);
    if (a == kUnmappedIndex)
        return;

    gDst.samples[a] = gSrc.samples[idx];
}
