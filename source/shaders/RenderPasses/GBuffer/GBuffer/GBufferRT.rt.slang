/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#include "Scene/SceneDefines.slangh"

import Scene.Raytracing;
import Scene.Intersection;
import GBufferRT;

ConstantBuffer<GBufferRT> gGBufferRT;

struct RayData
{
    // TODO: Passing in an empty payload struct doesn't work. Declare a dummy variable so that the compiler doesn't remove the declaration.
    int dummy;
};

//
// Shader entry point for miss shader.
//

[shader("miss")]
void miss(inout RayData rayData)
{
    // Write GBuffer data.
    uint2 launchIndex = DispatchRaysIndex().xy;
    gGBufferRT.writeMiss(launchIndex, WorldRayOrigin(), WorldRayDirection());
}

//
// Shader entry points for TriangleMesh hit groups.
//

[shader("anyhit")]
void anyHit(inout RayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    if (GBufferRT::kUseAlphaTest)
    {
        // Alpha test for non-opaque geometry.
        GeometryInstanceID instanceID = getGeometryInstanceID();
        VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
        const uint materialID = gScene.getMaterialID(instanceID);
        if (gScene.materials.alphaTest(v, materialID, 0.f))
            IgnoreHit();
    }
}

[shader("closesthit")]
void closestHit(inout RayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    TriangleHit triangleHit;
    triangleHit.instanceID = getGeometryInstanceID();
    triangleHit.primitiveIndex = PrimitiveIndex();
    triangleHit.barycentrics = attribs.barycentrics;

    // Write GBuffer data.
    uint2 launchIndex = DispatchRaysIndex().xy;
    gGBufferRT.writeHit(launchIndex, WorldRayOrigin(), WorldRayDirection(), HitInfo(triangleHit), RayTCurrent());
}

//
// Shader entry points for DisplacedTriangleMesh hit groups.
//

[shader("intersection")]
void displacedTriangleMeshIntersection()
{
    const Ray ray = Ray(WorldRayOrigin(), WorldRayDirection(), RayTMin(), RayTCurrent());
    DisplacedTriangleMeshIntersector::Attribs attribs;
    float t;
    if (DisplacedTriangleMeshIntersector::intersect(ray, getGeometryInstanceID(), PrimitiveIndex(), attribs, t))
    {
        ReportHit(t, 0, attribs);
    }
}

[shader("closesthit")]
void displacedTriangleMeshClosestHit(inout RayData rayData, DisplacedTriangleMeshIntersector::Attribs attribs)
{
    DisplacedTriangleHit displacedTriangleHit;
    displacedTriangleHit.instanceID = getGeometryInstanceID();
    displacedTriangleHit.primitiveIndex = PrimitiveIndex();
    displacedTriangleHit.barycentrics = attribs.barycentrics;
    displacedTriangleHit.displacement = attribs.displacement;

    // Write GBuffer data.
    uint2 launchIndex = DispatchRaysIndex().xy;
    gGBufferRT.writeHit(launchIndex, WorldRayOrigin(), WorldRayDirection(), HitInfo(displacedTriangleHit), RayTCurrent());
}

//
// Shader entry points for Curve hit groups.
//

[shader("intersection")]
void curveIntersection()
{
    const Ray ray = Ray(WorldRayOrigin(), WorldRayDirection(), RayTMin(), RayTCurrent());
    CurveIntersector::Attribs attribs;
    float t;
    if (CurveIntersector::intersect(ray, getGeometryInstanceID(), PrimitiveIndex(), attribs, t))
    {
        ReportHit(t, 0, attribs);
    }
}

[shader("closesthit")]
void curveClosestHit(inout RayData hitData, CurveIntersector::Attribs attribs)
{
    CurveHit curveHit;
    curveHit.instanceID = getGeometryInstanceID();
    curveHit.primitiveIndex = PrimitiveIndex();
    curveHit.barycentrics = attribs.barycentrics;

    // Write GBuffer data.
    uint2 launchIndex = DispatchRaysIndex().xy;
    gGBufferRT.writeHit(launchIndex, WorldRayOrigin(), WorldRayDirection(), HitInfo(curveHit), RayTCurrent());
}

//
// Shader entry points for SDFGrid hit groups.
//

[shader("intersection")]
void sdfGridIntersection()
{
    const Ray ray = Ray(WorldRayOrigin(), WorldRayDirection(), RayTMin(), RayTCurrent());
    SDFGridHitData sdfGridHitData;
    float t;
    if (SDFGridIntersector::intersect(ray, getGeometryInstanceID(), PrimitiveIndex(), sdfGridHitData, t))
    {
        ReportHit(t, 0, sdfGridHitData);
    }
}

[shader("closesthit")]
void sdfGridClosestHit(inout RayData hitData, SDFGridHitData sdfGridHitData)
{
    // Store hit information. Note we don't access the materials here.
    SDFGridHit sdfGridHit;
    sdfGridHit.instanceID = getGeometryInstanceID();
    sdfGridHit.hitData = sdfGridHitData;

    // Write GBuffer data.
    uint2 launchIndex = DispatchRaysIndex().xy;
    gGBufferRT.writeHit(launchIndex, WorldRayOrigin(), WorldRayDirection(), HitInfo(sdfGridHit), RayTCurrent());
}

//
// Shader entry point for ray generation shader.
//

[shader("raygeneration")]
void rayGen()
{
    uint2 launchIndex = DispatchRaysIndex().xy;

    GpuTimer timer;
    gGBufferRT.beginTime(timer);

    // Trace ray.
    const Ray ray = gGBufferRT.generateRay(launchIndex);
    RayData rayData;
    TraceRay(
        gScene.rtAccel,
        GBufferRT::kRayFlags,
        0xff /* instanceInclusionMask */,
        0 /* hitIdx */,
        rayTypeCount,
        0 /* missIdx */,
        ray.toRayDesc(),
        rayData
    );

    gGBufferRT.writeAux(launchIndex, ray);

    gGBufferRT.endTime(launchIndex, timer);
}
