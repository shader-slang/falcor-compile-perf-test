/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
import RenderPasses.Shared.Denoising.NRDConstants;
import InternalPathTracer;
import InternalPathTracerNRD;

import Utils.NVAPI; // TODO(@skallweit) this is currently needed to make g_NvidiaExt visible

// Pass a dummy payload to HitObject::TraceRay.
// This turns path state into contination spill and decrease performance.
#define USE_DUMMY_PAYLOAD 0

ParameterBlock<InternalPathTracer> gPathTracer;
ReorderTraceRayScheduler gScheduler;

// Define ray indices.
static const uint kRayTypeScatterGeneral = 0;
static const uint kRayTypeVisibility = 1;
static const uint kMissScatter = 0;
static const uint kMissVisibility = 1;

struct DummyPayload { uint dummy; };
struct DummyAttribs { uint dummy; };

typedef PackedPathState PathPayload;

struct VisibilityPayload
{
    bool visible;
};

struct ClosestHitQuery : IClosestHitQuery
{
    HitObject hitObject;

    [mutating] bool traceRay(inout PathState path, const Ray ray, out HitInfo hit, out float hitT)
    {
#if USE_DUMMY_PAYLOAD
        DummyPayload payload = {};
#else
        // Passing path state as the payload reduces local continuation spills and improves performance.
        PathPayload payload = path.encode();
#endif
        const uint rayFlags = kUseAlphaTest ? RAY_FLAG_NONE : RAY_FLAG_FORCE_OPAQUE;
        hitObject = HitObject::TraceRay(gScene.rtAccel, rayFlags, 0xff /* instanceInclusionMask */, kRayTypeScatterGeneral /* hitIdx */, rayTypeCount, kMissScatter, ray.toRayDesc(), payload);
#if !USE_DUMMY_PAYLOAD
        path.decode(payload);
#endif

        hit = makeHitInfo(hitObject);
        hitT = hitObject.GetRayDesc().TMax;
        return hit.isValid();
    }
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

/** Helper function to create a HitInfo from a HitObject.
*/
HitInfo makeHitInfo(const HitObject hitObject)
{
    HitInfo hit = {};

    if (hitObject.IsHit())
    {
        const GeometryInstanceID instanceID = GeometryInstanceID(hitObject.GetInstanceID(), hitObject.GetGeometryIndex());
        const uint primitiveIndex = hitObject.GetPrimitiveIndex();

        // Lookup geometry type as we cannot reconstruct that from the HitObject directly.
        const GeometryType type = gScene.getGeometryInstanceType(instanceID);

        switch (type)
        {
        case GeometryType::None:
            break;
        case GeometryType::TriangleMesh:
        {
            TriangleHit triangleHit;
            triangleHit.instanceID = instanceID;
            triangleHit.primitiveIndex = primitiveIndex;
            BuiltInTriangleIntersectionAttributes attribs = hitObject.GetAttributes<BuiltInTriangleIntersectionAttributes>();
            triangleHit.barycentrics = attribs.barycentrics;
            hit = HitInfo(triangleHit);
            break;
        }
        case GeometryType::DisplacedTriangleMesh:
        {
            DisplacedTriangleHit displacedTriangleHit;
            displacedTriangleHit.instanceID = instanceID;
            displacedTriangleHit.primitiveIndex = primitiveIndex;
            DisplacedTriangleMeshIntersector.Attribs attribs = hitObject.GetAttributes<DisplacedTriangleMeshIntersector.Attribs>();
            displacedTriangleHit.barycentrics = attribs.barycentrics;
            displacedTriangleHit.displacement = attribs.displacement;
            hit = HitInfo(displacedTriangleHit);
            break;
        }
        case GeometryType::Curve:
        {
            CurveHit curveHit;
            curveHit.instanceID = instanceID;
            curveHit.primitiveIndex = primitiveIndex;
            CurveIntersector.Attribs attribs = hitObject.GetAttributes<CurveIntersector.Attribs>();
            curveHit.barycentrics = attribs.barycentrics;
            hit = HitInfo(curveHit);
            break;
        }
#ifdef FALCOR_INTERNAL
        case GeometryType::CurveOTS:
        {
            CurveOTSHit curveOTSHit;
            curveOTSHit.instanceID = instanceID;
            curveOTSHit.primitiveIndex = primitiveIndex;
            BuiltInTriangleIntersectionAttributes attribs = hitObject.GetAttributes<BuiltInTriangleIntersectionAttributes>();
            curveOTSHit.barycentrics = attribs.barycentrics;
            hit = HitInfo(curveOTSHit);
            break;
        }
#endif // FALCOR_INTERNAL
        case GeometryType::SDFGrid:
        {
            SDFGridHit sdfGridHit;
            sdfGridHit.instanceID = instanceID;
            SDFGridHitData attribs = hitObject.GetAttributes<SDFGridHitData>();
            sdfGridHit.hitData = attribs;
            hit = HitInfo(sdfGridHit);
            break;
        }
        case GeometryType::Custom:
            // Unsupported.
            break;
        }
    }

    return hit;
}

/** Reordering scheduler using SER/HitObject API.

    The API allows to implement the IClosestHitQuery interface, because
    after executing ray traversal using HitObject::TraceRay(), control is given back to the
    caller before the CHS/MS is explicitly invoked using InvokeHitObject().
    This allows for executing volume distance sampling as part of PathTracer::nextHit.
*/
struct ReorderTraceRayScheduler
{
    /** Handle a miss.
        Hit information is encoded in the payload.
        Note: This is also called for volume hits.
        \param[in,out] payload Payload data.
    */
    void handleMiss(inout PathPayload payload)
    {
        PathState path;
        path.decode(payload);

        gPathTracer.setupPathLogging(path);

        // Volume hits invoke the miss shader, so we need to check here.
        if (path.isHit() && path.hit.getType() == HitType::Volume)
        {
            VisibilityQuery vq;
            gPathTracer.handleVolumeHit(path, vq);
        }
        else
        {
            gPathTracer.handleMiss(path);
        }

        payload = path.encode();
    }

    /** Handle a hit.
        \param[in,out] payload Payload data.
    */
    void handleHit(inout PathPayload payload)
    {
        PathState path;
        path.decode(payload);

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

    /** Traces a path and write result to output.
        \param[in] pathID Path ID.
    */
    void tracePath(uint pathID)
    {
        PathState path = {};

        gPathTracer.generatePath(pathID, path);
        gPathTracer.setupPathLogging(path);

        if (path.isHit())
        {
            // Handle primary hits.

            // Create a HitObject from a HitInfo.
            // The created HitObject does only contain the minimal required data,
            // most data is already passed in the path.hit object.
            // For volume hits, a miss object is returned(and handled in the handleMiss() function).
            // TODO: Move to a utility function when Slang limitation has been resolved. See https://gitlab-master.nvidia.com/nvresearch-gfx/Tools/Falcor/-/issues/1611
            HitObject hitObject;
            RayDesc dummyRay = {};
            DummyAttribs dummyAttribs = {};

            switch (path.hit.getType())
            {
            case HitType::None:
                hitObject = HitObject::MakeMiss(kMissScatter, dummyRay);
                break;
            case HitType::Volume:
                hitObject = HitObject::MakeMiss(kMissScatter, dummyRay);
                break;
            default:
            {
                const GeometryInstanceID instanceID = path.hit.getInstanceID();
                const GeometryInstanceData instance = gScene.getGeometryInstance(instanceID);
                const uint primitiveIndex = path.hit.getPrimitiveIndex();
                hitObject = HitObject::MakeHit(
                    gScene.rtAccel,         // AccelerationStructure
                    instance.instanceIndex, // InstanceIndex
                    instance.geometryIndex, // GeometryIndex
                    primitiveIndex,         // PrimitiveIndex
                    0,                      // HitKind
                    kRayTypeScatterGeneral, // RayContributionToHitGroupIndex
                    rayTypeCount,           // MultiplierForGeometryContributionToHitGroupIndex
                    dummyRay,               // Ray
                    dummyAttribs            // Attributes
                );
                break;
            }
            }

            PathPayload payload = path.encode();
            HitObject::Invoke(gScene.rtAccel, hitObject, payload);
            path.decode(payload);
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
            PathPayload payload = {};
            HitObject hitObject;

#if USE_REORDER_TRACE_RAY_FAST_PATH
            {
                // Advance to next path vertex.
                path.incrementVertexIndex();

                // Trace ray.
                logTraceRay(PixelStatsRayType::ClosestHit);
                const Ray ray = path.getScatterRay();
                payload = path.encode();
                const uint rayFlags = kUseAlphaTest ? RAY_FLAG_NONE : RAY_FLAG_FORCE_OPAQUE;
                hitObject = HitObject::TraceRay(gScene.rtAccel, rayFlags, 0xff /* instanceInclusionMask */, kRayTypeScatterGeneral /* hitIdx */, rayTypeCount, kMissScatter, ray.toRayDesc(), payload);
                path.decode(payload);

                if (hitObject.IsHit())
                {
                    path.setHit(makeHitInfo(hitObject));
                    path.sceneLength += float16_t(hitObject.GetRayDesc().TMax);
                }
                else
                {
                    path.clearHit();
                    path.sceneLength = float16_t(kNRDInvalidPathLength);
                }
            }
#else
            // Use ClosestHitQuery abstraction instead of using HitObject::TraceRay directly.
            // This enables support for sampling volumes but also leads to continuation spill of
            // the volume sampling state which in turn decrease performance.
            ClosestHitQuery chq;
            gPathTracer.nextHit(path, chq);
            hitObject = chq.hitObject;
#endif

            // Add path coherence hints.
            const uint materialID = gScene.getMaterialID(path.hit.getInstanceID());
            const MaterialHeader materialHeader = gScene.materials.getMaterialHeader(materialID);
            const PathCoherenceHints pathHints = gPathTracer.getCoherenceHints(path, materialHeader.isDeltaSpecular(), materialHeader.isEmissive());
            // MSB = highest priority hint.
            uint coherenceHint =
                (pathHints.computesEmissive ? 0x1 : 0x0) |
                (pathHints.samplesLight ? 0x2 : 0x0) |
                (pathHints.terminated ? 0x4 : 0x0) |
                (pathHints.terminatedByRussianRoulette ? 0x8 : 0x0);

            // Reorder and invoke closest hit or miss shader.
            ReorderThread(hitObject, coherenceHint, 4);
            payload = path.encode();
            HitObject::Invoke(gScene.rtAccel, hitObject, payload);
            path.decode(payload);
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

            // Use SER to compact active threads.
            if (kSamplesPerPixel == 0)
            {
                bool needReorder = WaveActiveAnyTrue(samplesRemaining == 0);
                if (samplesRemaining == 0) break;
                if (needReorder)
                {
                    HitObject hitObject = HitObject::MakeNop();
                    ReorderThread(hitObject);
                }
            }
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
    gScheduler.handleHit(payload);
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
void scatterDisplacedTriangleMeshClosestHit(inout PathPayload payload : SV_RayPayload, DummyAttribs attribs : SV_IntersectionAttributes)
{
    gScheduler.handleHit(payload);
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
void scatterCurveClosestHit(inout PathPayload payload : SV_RayPayload, DummyAttribs attribs : SV_IntersectionAttributes)
{
    gScheduler.handleHit(payload);
}

//
// Shader entry points for CurveOTS hit groups.
//

[shader("closesthit")]
void scatterCurveOTSClosestHit(inout PathPayload payload : SV_RayPayload, DummyAttribs attribs : SV_IntersectionAttributes)
{
    gScheduler.handleHit(payload);
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
void scatterSdfGridClosestHit(inout PathPayload payload : SV_RayPayload, DummyAttribs attribs : SV_IntersectionAttributes)
{
    gScheduler.handleHit(payload);
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
