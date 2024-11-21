/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import Utils.Geometry.GeometryHelpers;

cbuffer CB
{
    uint n;
};

struct AABB2D
{
    float2 minPos;
    float2 maxPos;
};

StructuredBuffer<float3> pos;
StructuredBuffer<float3> normal;
StructuredBuffer<AABB2D> aabb;
RWStructuredBuffer<float3> result;

[numthreads(256, 1, 1)]
void testComputeRayOrigin(uint3 threadId: SV_DispatchThreadID)
{
    uint i = threadId.x;
    if (i >= n)
        return;

    result[i] = computeRayOrigin(pos[i], normal[i]);
}

Buffer<float4> spheres; // xyz: center, w: radius
RWStructuredBuffer<float> sinTheta, cosTheta;

[numthreads(256, 1, 1)]
void testBoundSphereAngle(uint3 threadId: SV_DispatchThreadID)
{
    const uint index = threadId.x;
    float sqrRadius = spheres[index].w * spheres[index].w;
    boundSphereSubtendedConeAngle(spheres[index].xyz, sqrRadius, sinTheta[index], cosTheta[index]);
}

Buffer<float3> origin, aabbMin, aabbMax;
RWStructuredBuffer<float3> coneDir;

[numthreads(256, 1, 1)]
void testBoundingConeAngleAverage(uint3 threadId: SV_DispatchThreadID)
{
    const uint index = threadId.x;
    boundBoxSubtendedConeAngleAverage(origin[index], aabbMin[index], aabbMax[index], coneDir[index], sinTheta[index], cosTheta[index]);
}

[numthreads(256, 1, 1)]
void testBoundingConeAngleCenter(uint3 threadId: SV_DispatchThreadID)
{
    const uint index = threadId.x;
    boundBoxSubtendedConeAngleCenter(origin[index], aabbMin[index], aabbMax[index], coneDir[index], sinTheta[index], cosTheta[index]);
}

[numthreads(256, 1, 1)]
void testComputeClippedTriangleArea2D(uint3 threadId: SV_DispatchThreadID)
{
    uint i = threadId.x;
    if (i >= n)
        return;

    float2 p[3];
    p[0] = pos[3 * i + 0].xy;
    p[1] = pos[3 * i + 1].xy;
    p[2] = pos[3 * i + 2].xy;

    float area = computeClippedTriangleArea2D(p, aabb[i].minPos, aabb[i].maxPos);
    result[i] = float3(area, 0, 0);
}
