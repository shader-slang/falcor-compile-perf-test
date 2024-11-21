/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import Utils.Geometry.IntersectionHelpers;

StructuredBuffer<float3> sphereCenter;
StructuredBuffer<float> sphereRadius;
StructuredBuffer<float3> rayOrigin;
StructuredBuffer<float3> rayDir;
RWStructuredBuffer<uint> isectResult;
RWStructuredBuffer<float3> isectLoc;

cbuffer TestCB
{
    int resultSize;
};

[numthreads(1, 1, 1)]
void testRaySphereIntersection()
{
    for (int i = 0; i < resultSize; ++i)
    {
        float t;
        if (intersectRaySphere(rayOrigin[i], rayDir[i], sphereCenter[i], sphereRadius[i], t))
        {
            isectResult[i] = 1;
            isectLoc[i] = rayOrigin[i] + t * rayDir[i];
        }
        else
        {
            isectResult[i] = 0;
        }
    }
}
