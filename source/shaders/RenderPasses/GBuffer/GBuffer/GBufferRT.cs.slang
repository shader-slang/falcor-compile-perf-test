/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import Scene.RaytracingInline;
import GBufferRT;

ConstantBuffer<GBufferRT> gGBufferRT;

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    if (any(pixel >= gGBufferRT.frameDim))
        return;

    GpuTimer timer;
    gGBufferRT.beginTime(timer);

    const Ray ray = gGBufferRT.generateRay(pixel);

    SceneRayQuery<GBufferRT::kUseAlphaTest> sceneRayQuery;
    HitInfo hit;
    float hitT;
    if (sceneRayQuery.traceRay(ray, hit, hitT, GBufferRT::kRayFlags, 0xff))
    {
        gGBufferRT.writeHit(pixel, ray.origin, ray.dir, hit, hitT);
    }
    else
    {
        gGBufferRT.writeMiss(pixel, ray.origin, ray.dir);
    }

    gGBufferRT.writeAux(pixel, ray);

    gGBufferRT.endTime(pixel, timer);
}
