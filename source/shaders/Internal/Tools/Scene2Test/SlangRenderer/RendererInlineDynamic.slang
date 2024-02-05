/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
import Internal.Scene2.SharedTypes;
import Internal.Scene2.Cameras.Camera;
import Internal.Scene2.Scene2;
import Internal.Scene2.HitInfo;
import Internal.Scene2.Geometry.GeoBase;
import Internal.Scene2.Geometry.GeoAccessors;
import Internal.Scene2.VertexData;
import Internal.Tools.Scene2Test.SlangRenderer.TrivialPathTracer;

import Utils.Math.Ray;

struct RendererInlineDynamic
{
    bool isInline() { return true; }

    IHitInfo traceRayNearestHit<let Flags : int>(Ray ray)
    {
        RayQuery<Flags> q;
        let rayFlags = RAY_FLAG_NONE;
        float t = ray.tMax;
        GeoIsectAttributes geoIsectAttr;

        q.TraceRayInline(gScene.tlas, rayFlags, 0xff, ray.toRayDesc());

        while (q.Proceed())
        {
            if (q.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE)
            {
                // Slang doesn't seem to codegen the actual body of getGeometryData
                // GeometryType2 geoType = gScene.getGeometryData(q.CandidateInstanceID(), q.CandidateInstanceIndex(),
                // q.CandidateGeometryIndex()).geoType;
                GeometryType2 geoType = gScene.geometryData[q.CandidateInstanceID() + q.CandidateGeometryIndex()].geoType;
                IGeoIntersector geoIntersector = gScene.getGeoIntersector(geoType);
                if (geoIntersector.intersect(q, geoIsectAttr, t))
                    q.CommitProceduralPrimitiveHit(t);
            }
        }

        InvalidHitInfo invalidHit;
        if (q.CommittedStatus() == COMMITTED_NOTHING)
            return invalidHit;

        // const GeometryType2 geoType = gScene.getGeometryData(q.CommittedInstanceID(), q.CommittedInstanceIndex(),
        // q.CommittedGeometryIndex()).geoType;
        GeometryType2 geoType = gScene.geometryData[q.CommittedInstanceID() + q.CommittedGeometryIndex()].geoType;
        IGeoIntersector geoIntersector = gScene.getGeoIntersector(geoType);
        return geoIntersector.makeHitInfo(q, geoIsectAttr);
    }

    typedef TrivialPathTracer::PathState Payload;

    void handleHit<THitInfo : IHitInfo>(inout Payload payload, const THitInfo hitInfo) { gPathTracer.handleHit(payload, hitInfo); }

    void handleMiss(inout Payload payload) { gPathTracer.handleMiss(payload); }

    // returns true when all ok, false otherwise
    bool checkBackfacing(Ray ray)
    {
        float backFacingT = 1e36f;
        IHitInfo hit = traceRayNearestHit<RAY_FLAG_CULL_FRONT_FACING_TRIANGLES>(ray);
        if (hit.isValid())
        {
            VertexData v;
            if (!hit.getVertexData(ray, v))
                return true;
            backFacingT = dot(ray.dir, v.posW - ray.origin);
        }
        else
            return true; // hit no backface, all is well

        // At this stage, we hit a backface, and now we want to make sure
        // there is a front face that's closer

        hit = traceRayNearestHit<RAY_FLAG_CULL_BACK_FACING_TRIANGLES>(ray);
        if (!hit.isValid())
            return false; // haven't hit front facing, it is a problem
        VertexData v;
        if (!hit.getVertexData(ray, v))
            return false;
        float frontFacingT = dot(ray.dir, v.posW - ray.origin);
        return frontFacingT < backFacingT;
    }

    float3 tracePrimary(Ray ray, uint seed)
    {
        // if (!checkBackfacing(ray))
        //     return float3(1, 0, 1);

        Payload payload = Payload(ray, seed);
        for (int depth = 0; depth < 2; ++depth)
        {
            if (payload.isActive == 0)
                break;
            IHitInfo hit = traceRayNearestHit<RAY_FLAG_FORCE_OPAQUE>(payload.ray);
            if (hit.isValid())
                handleHit(payload, hit);
            else
                handleMiss(payload);
        }
        return payload.totalContribution;
    }
}
