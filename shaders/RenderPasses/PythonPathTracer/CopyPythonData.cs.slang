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
    PythonRayState gState;
    StructuredBuffer<float3> gWo;
    StructuredBuffer<float> gPdf;
    StructuredBuffer<float3> gWeight;
    bool gIsSampleData;
}

[numthreads(256, 1, 1)]
void copy(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    const uint idx = dispatchThreadId.x;
    if (idx >= gState.elemCount)
        return;

    MaterialSample ms = gState.samples[idx];
    if (gIsSampleData)
    {
        ms.setWo(gWo[idx]);
    }
    else
    {
        ms.setWeight(gWeight[idx]);
        ms.setPdf(gPdf[idx]);
    }
    gState.samples[idx] = ms;
}
