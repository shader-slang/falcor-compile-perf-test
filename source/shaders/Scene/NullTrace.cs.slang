/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import Utils.Attributes;

[root] RaytracingAccelerationStructure gTlas;
RWTexture2D<uint> gOutput;

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    RayDesc ray;
    ray.Origin = float3(0, 0, 0);
    ray.Direction = float3(1, 0, 0);
    ray.TMin = 0.f;
    ray.TMax = 1.f;

    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> rayQuery;
    rayQuery.TraceRayInline(gTlas, RAY_FLAG_NONE, 0xff, ray);
    rayQuery.Proceed();
    gOutput[dispatchThreadId.xy] = (rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT) ? 1 : 0;
}
