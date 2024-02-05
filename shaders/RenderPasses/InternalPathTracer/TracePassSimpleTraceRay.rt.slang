/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
#include "Scene/SceneDefines.slangh"

import Scene.Raytracing;
import Scene.RaytracingInline; // For visibility queries.
import Scene.Intersection;
import Rendering.Utils.PixelStats;
import Utils.Math.PackedFormats;
import Utils.Math.FormatConversion;
import InternalPathTracer;
import InternalPathTracerNRD;

ParameterBlock<InternalPathTracer> gPathTracer;
SimpleTraceRayScheduler gScheduler;

// Define ray indices.
static const uint kRayTypeScatter = 0;
static const uint kRayTypeVisibility = 1;
static const uint kMissScatter = 0;
static const uint kMissVisibility = 1;

typedef PackedPathState PathPayload;

struct VisibilityPayload
{
    bool visible;
};

struct VisibilityQuery : IVisibilityQuery
{
    bool traceVisibilityRay(const Ray ray)
    {
#if USE_TRACE_RAY_VISIBILITY
        VisibilityPayload payload = { false }; // Set to true in miss shader.
        uint rayFlags = RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;
        if (!kUseAlphaTest) rayFlags |= RAY_FLAG_FORCE_OPAQUE;
        TraceRay(gScene.rtAccel, rayFlags, 0xff /* instanceInclusionMask */, kRayTypeVisibility /* hitIdx */, rayTypeCount, kMissVisibility /* missIdx */, ray.toRayDesc(), payload);
        return payload.visible;
#else
        SceneRayQuery<kUseAlphaTest> sceneRayQuery;
        return sceneRayQuery.traceVisibilityRay(ray, RAY_FLAG_NONE, 0xff);
#endif
    }
};

/** Simple scheduler (fullscreen raytracing pass) using DXR 1.0 TraceRay.

    The TraceRay call directly dispatches into a closest hit / miss shader.
    This makes it hard to implement the IClosestHitQuery interface
    unless we would use the CHS/MS to simply return the hit information.
    But we want CHS/MS to do the actual handling of the hit or miss (shading).
    For that reason, we implement a custom `nextHit` that replaces `InternalPathTracer::nextHit`
    but without support for volume sampling.
*/
struct SimpleTraceRayScheduler
{
    /** Handle a miss.
        \param[in,out] payload Payload data.
    */
    void handleMiss(inout PathPayload payload)
    {
        PathState path;
        path.decode(payload);
        path.clearHit();

        gPathTracer.setupPathLogging(path);
        gPathTracer.handleMiss(path);

        payload = path.encode();
    }

    /** Handle a hit.
        \param[in,out] payload Payload data.
        \param[in] hit Hit info.
    */
    void handleHit(inout PathPayload payload, const HitInfo hit)
    {
        PathState path;
        path.decode(payload);
        path.setHit(hit);

        gPathTracer.setupPathLogging(path);
        VisibilityQuery vq;
#if defined(DELTA_REFLECTION_PASS)
        gPathTracer.handleDeltaReflectionHit(path);
#elif defined(DELTA_TRANSMISSION_PASS)
        gPathTracer.handleDeltaTransmissionHit(path);
#else
        gPathTracer.handleHit(path, vq);
#endif

        payload = path.encode();
    }

    /** Trace a ray to find the next hit.
        This replaces `InternalPathTracer::nextHit` but without support for volume sampling.
        \param[in,out] path Path state.
    */
    void nextHit(inout PathState path)
    {
        // Advance to next path vertex.
        path.incrementVertexIndex();

        // Trace ray.
        logTraceRay(PixelStatsRayType::ClosestHit);
        const Ray ray = path.getScatterRay();
        PathPayload payload = path.encode();
        uint rayFlags = RAY_FLAG_NONE;
        if (!kUseAlphaTest) rayFlags |= RAY_FLAG_FORCE_OPAQUE;
        TraceRay(gScene.rtAccel, rayFlags, 0xff /* instanceInclusionMask */, kRayTypeScatter /* hitIdx */, rayTypeCount, kMissScatter /* missIdx */, ray.toRayDesc(), payload);

        path.decode(payload);
    }

    /** Traces a path and write result to output.
        \param[in] pathID Path ID.
    */
    void tracePath(uint pathID)
    {
        PathState path = {};

        gPathTracer.generatePath(pathID, path);
        gPathTracer.setupPathLogging(path);

        // Handle primary hit/miss.
        if (path.isHit())
        {
            VisibilityQuery vq;
#if defined(DELTA_REFLECTION_PASS)
            gPathTracer.handleDeltaReflectionHit(path);
#elif defined(DELTA_TRANSMISSION_PASS)
            gPathTracer.handleDeltaTransmissionHit(path);
#else
            gPathTracer.handleHit(path, vq);
#endif
        }
        else
        {
            // Note the primary miss has already been handled by the separate path generation pass
            // the path tracer runs first. Abort execution here to avoid double work.
            return;
        }

        // Trace path.
        while (path.isActive())
        {
            nextHit(path);
        }

#if !defined(DELTA_REFLECTION_PASS) && !defined(DELTA_TRANSMISSION_PASS)
        gPathTracer.writeOutput(path);
#endif
    }

    /** Runs the path tracer for a single pixel.
        \param[in] pixel Pixel index.
    */
    void run(uint2 pixel)
    {
        // Determine number of samples to take.
        uint samplesRemaining = kSamplesPerPixel;
        if (kSamplesPerPixel == 0)
        {
            samplesRemaining = gPathTracer.sampleCount[pixel];
        }

        // Loop over samples.
        while (samplesRemaining > 0)
        {
            samplesRemaining -= 1;
            uint pathID = pixel.x | (pixel.y << 12) | (samplesRemaining << 24);
            tracePath(pathID);
        }
    }
}

//
// Shader entry point for miss shaders.
//

[shader("miss")]
void scatterMiss(inout PathPayload payload : SV_RayPayload)
{
    gScheduler.handleMiss(payload);
}

[shader("miss")]
void visibilityMiss(inout VisibilityPayload payload : SV_RayPayload)
{
    payload.visible = true;
}

//
// Shader entry points for TriangleMesh hit groups.
//

[shader("anyhit")]
void scatterTriangleAnyHit(inout PathPayload payload : SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes)
{
#if USE_ALPHA_TEST
    // Alpha test for non-opaque geometry.
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);
    if (gScene.materials.alphaTest(v, materialID, 0.f)) IgnoreHit();
#endif
}

[shader("closesthit")]
void scatterTriangleClosestHit(inout PathPayload payload : SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes)
{
    TriangleHit triangleHit;
    triangleHit.instanceID = getGeometryInstanceID();
    triangleHit.primitiveIndex = PrimitiveIndex();
    triangleHit.barycentrics = attribs.barycentrics;
    gScheduler.handleHit(payload, HitInfo(triangleHit));
}

[shader("anyhit")]
void visibilityTriangleAnyHit(inout VisibilityPayload payload : SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes)
{
#if USE_ALPHA_TEST
    // Alpha test for non-opaque geometry.
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);
    if (gScene.materials.alphaTest(v, materialID, 0.f)) IgnoreHit();
#endif
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
void scatterDisplacedTriangleMeshClosestHit(inout PathPayload payload, DisplacedTriangleMeshIntersector::Attribs attribs)
{
    DisplacedTriangleHit displacedTriangleHit;
    displacedTriangleHit.instanceID = getGeometryInstanceID();
    displacedTriangleHit.primitiveIndex = PrimitiveIndex();
    displacedTriangleHit.barycentrics = attribs.barycentrics;
    displacedTriangleHit.displacement = attribs.displacement;
    gScheduler.handleHit(payload, HitInfo(displacedTriangleHit));
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
void scatterCurveClosestHit(inout PathPayload payload, in CurveIntersector::Attribs attribs)
{
    CurveHit curveHit;
    curveHit.instanceID = getGeometryInstanceID();
    curveHit.primitiveIndex = PrimitiveIndex();
    curveHit.barycentrics = attribs.barycentrics;
    gScheduler.handleHit(payload, HitInfo(curveHit));
}

//
// Shader entry points for CurveOTS hit groups.
//

[shader("closesthit")]
void scatterCurveOTSClosestHit(inout PathPayload payload : SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes)
{
    CurveOTSHit curveOTSHit;
    curveOTSHit.instanceID = getGeometryInstanceID();
    curveOTSHit.primitiveIndex = PrimitiveIndex();
    curveOTSHit.barycentrics = attribs.barycentrics;
    gScheduler.handleHit(payload, HitInfo(curveOTSHit));
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
void scatterSdfGridClosestHit(inout PathPayload payload, in SDFGridHitData sdfGridHitData)
{
    SDFGridHit sdfGridHit;
    sdfGridHit.instanceID = getGeometryInstanceID();
    sdfGridHit.hitData = sdfGridHitData;
    gScheduler.handleHit(payload, HitInfo(sdfGridHit));
}

//
// Shader entry point for ray generation shader.
//

[shader("raygeneration")]
void rayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 frameDim = DispatchRaysDimensions().xy;
    if (all(pixel >= frameDim)) return;

    gScheduler.run(pixel);
}
