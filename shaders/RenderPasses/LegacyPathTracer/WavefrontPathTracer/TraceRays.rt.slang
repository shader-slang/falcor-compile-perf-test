/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/

/** Trace kernels for shadow and scatter rays.

    The resources bound are:

    Global:
    - 1 acceleration structure
    - 1 buffer for the path state.
    - 1 buffer for the SampleGenerator (if Sobol).
    - buffers for the vertex/geometry data (for alpha testing)

    Local:
    - 1 cbuffer (4B) per hit geometry group.
*/
#include "Utils/Math/MathConstants.slangh"

import Scene.Raytracing;
import Utils.Geometry.GeometryHelpers;
import Utils.Math.PackedFormats;
import Utils.Debug.PixelDebug;
import Rendering.Materials.InteriorListHelpers;
import Rendering.Utils.PixelStats;
import RenderPasses.LegacyPathTracer.PathTracer.PathTracerParams;
import RenderPasses.LegacyPathTracer.WavefrontPathTracer.PathState;

static const float kRayTMax = FLT_MAX;
static const bool kUnconditionalTraceRay = UNCONDITIONAL_TRACE_RAY;
static const bool kForceOpaque = !kUseAlphaTest;

RWBuffer<uint> gPathFlags;                  ///< Path flags indicating which rays are active.
RWStructuredBuffer<PackedHitInfo> gPathHit; ///< Path hit info for scatter ray.
StructuredBuffer<PathRays> gPathRays;       ///< The path ray parameters (read only).
StructuredBuffer<float3> gPathTransmission; ///< The transmission ray origins.
RWStructuredBuffer<InteriorList> gPathInteriorList;

/************************** Ray index 0: Scatter ray ************************ */

/** Payload for scatter ray (8-12B).
 */
struct ScatterRayData
{
    PackedHitInfo packedHitInfo; ///< Packed HitInfo data, or kInvalidIndex in the first component if ray missed.
};

// TODO: The API is supposed to support disabling the miss shader, but Falcor seems to require a no-op shader for now.
// See https://gitlab-master.nvidia.com/nvresearch-gfx/Tools/Falcor/issues/428
[shader("miss")]
void scatterMiss(inout ScatterRayData rayData: SV_RayPayload) {}

[shader("anyhit")]
void scatterAnyHit(inout ScatterRayData rayData: SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs: SV_IntersectionAttributes)
{
#if USE_ALPHA_TEST
    // Alpha test for non-opaque geometry.
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);
    if (gScene.materials.alphaTest(v, materialID, 0.f))
        IgnoreHit();
#endif
}

[shader("closesthit")]
void scatterClosestHit(
    inout ScatterRayData rayData: SV_RayPayload,
    BuiltInTriangleIntersectionAttributes attribs: SV_IntersectionAttributes
)
{
    // Store hit information. Note we don't access the materials here.
    TriangleHit triangleHit;
    triangleHit.instanceID = getGeometryInstanceID();
    triangleHit.primitiveIndex = PrimitiveIndex();
    triangleHit.barycentrics = attribs.barycentrics;
    rayData.packedHitInfo = triangleHit.pack();
}

/************************** Ray index 1: Shadow ray ************************ */

/** Payload for shadow ray (4B).
 */
struct ShadowRayData
{
    bool visible;
};

[shader("miss")]
void shadowMiss(inout ShadowRayData rayData: SV_RayPayload)
{
    rayData.visible = true;
}

[shader("anyhit")]
void shadowAnyHit(inout ShadowRayData rayData: SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs: SV_IntersectionAttributes)
{
#if USE_ALPHA_TEST
    // Alpha test for non-opaque geometry.
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);
    if (gScene.materials.alphaTest(v, materialID, 0.f))
        IgnoreHit();
#endif
}

/** ******************************** RayGen ******************************** */

/** Entry point for trace kernel, tracing shadow and scatter rays.
 */
[shader("raygeneration")]
void rayGen()
{
    printSetPixel(DispatchRaysIndex().xy);
    logSetPixel(DispatchRaysIndex().xy);

    const uint idx = getIdx(DispatchRaysIndex());
    if (idx == kInvalidIndex)
        return;

    // Load the path flags to see which rays we need to trace.
    // Note the flags are encoded in the low 24 bits, we leave the high bits unchanged.
    uint flags = gPathFlags[idx];

    bool traceShadowRays = (flags & uint(PathFlags::anyShadowRays));
    bool traceScatterRay = (flags & uint(PathFlags::scatterRay));
    if (!(traceShadowRays || traceScatterRay))
        return;

    // We need to trace the shadow and/or scatter rays. Load the ray origin and proceed.
    float3 origin = gPathRays[idx].origin;

    for (uint i = 0; i < kLightSamplesPerVertex; i++)
    {
        bool traceShadowRay = (flags & (uint(PathFlags::shadowRay) << i));

        if (kUnconditionalTraceRay || traceShadowRay)
        {
            RayDesc ray;
            ray.Origin = origin;
            ray.Direction = float3(1, 0, 0); // TODO: Is nonzero dir required for dummy rays?
            ray.TMin = 0.f;
            ray.TMax = 0.f;

            if (traceShadowRay)
            {
                float4 rayParams = gPathRays[idx].shadowRay[i]; // Direction (xyz) and distance (w).
                ray.Direction = rayParams.xyz;
                ray.TMax = rayParams.w;
            }

            // Trace the ray.
            ShadowRayData shadowRayData;
            shadowRayData.visible = false;

            uint rayFlags = RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;
            if (kForceOpaque)
                rayFlags |= RAY_FLAG_FORCE_OPAQUE;
            TraceRay(
                gScene.rtAccel,
                rayFlags,
                0xff /* instanceInclusionMask */,
                kRayTypeShadow /* hitIdx */,
                rayTypeCount,
                kRayTypeShadow /* missIdx */,
                ray,
                shadowRayData
            );

            if (traceShadowRay)
                logTraceRay(PixelStatsRayType::Visibility);

            if (!shadowRayData.visible)
            {
                flags &= ~(uint(PathFlags::shadowRay) << i); // Clear shadow ray bit
            }
        }
    }

    if (kUnconditionalTraceRay || traceScatterRay)
    {
        RayDesc ray;
        ray.Origin = origin;
        ray.Direction = float3(1, 0, 0); // TODO: Is nonzero dir required for dummy rays?
        ray.TMin = 0.f;
        ray.TMax = 0.f;

        if (traceScatterRay)
        {
            if (flags & uint(PathFlags::transmission))
                origin = gPathTransmission[idx];
            float3 dir = gPathRays[idx].getScatterDir();

            ray.Origin = origin;
            ray.Direction = dir;
            ray.TMin = 0.f;
            ray.TMax = kRayTMax;
        }

        // Trace the ray.
        ScatterRayData scatterRayData = {};
        uint rejectedHits = 0;

        // For nested dielectrics, we potentially have to trace additional rays after false intersections.
        while (true)
        {
            scatterRayData.packedHitInfo.x = 0; // Clear hit

            uint rayFlags = RAY_FLAG_NONE;
            if (kForceOpaque)
                rayFlags |= RAY_FLAG_FORCE_OPAQUE;
            TraceRay(
                gScene.rtAccel,
                rayFlags,
                0xff /* instanceInclusionMask */,
                kRayTypeScatter /* hitIdx */,
                rayTypeCount,
                kRayTypeScatter /* missIdx */,
                ray,
                scatterRayData
            );

            if (traceScatterRay)
            {
                logTraceRay(PixelStatsRayType::ClosestHit);

                // Check for false intersections.
                if (kUseNestedDielectrics && scatterRayData.packedHitInfo.x != 0)
                {
                    const TriangleHit triangleHit = TriangleHit(scatterRayData.packedHitInfo);
                    uint materialID = gScene.getMaterialID(triangleHit.instanceID);
                    uint nestedPriority = gScene.materials.getMaterialHeader(materialID).getNestedPriority();
                    if (!gPathInteriorList[idx].isTrueIntersection(nestedPriority))
                    {
                        if (rejectedHits < kMaxRejectedHits)
                        {
                            rejectedHits++;
                            VertexData v = gScene.getVertexData(triangleHit);
                            bool frontFacing = dot(-ray.Direction, v.faceNormalW) >= 0.f;
                            gPathInteriorList[idx].handleIntersection(materialID, nestedPriority, frontFacing);
                            ray.Origin = computeRayOrigin(v.posW, frontFacing ? -v.faceNormalW : v.faceNormalW);
                            continue;
                        }
                        else
                        {
                            // Terminate path.
                            scatterRayData.packedHitInfo.x = 0;
                            flags = 0;
                        }
                    }
                }
            }

            break;
        }

        // Store hit info.
        if (traceScatterRay && scatterRayData.packedHitInfo.x != 0)
        {
            flags |= uint(PathFlags::scatterHit);
            gPathHit[idx] = scatterRayData.packedHitInfo;
        }
    }

    gPathFlags[idx] = flags;
}
