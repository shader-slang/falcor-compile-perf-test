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
import Internal.Scene2.VertexData;
import Internal.Tools.Scene2Test.SlangRenderer.TrivialPathTracer;

import Utils.Math.Ray;

struct ShaderRtDelegate
{
    typedef TrivialPathTracer::PathState Payload;

    struct VisibilityPayload
    {
        bool visible = { false }; // Set to true in miss shader.
    };

    bool isInline() { return false; }

    void handleHit<THitInfo : IHitInfo>(inout Payload payload, const THitInfo hitInfo) { gPathTracer.handleHit(payload, hitInfo); }

    void handleMiss(inout Payload payload) { gPathTracer.handleMiss(payload); }

    float3 tracePrimarySER(Ray ray, uint seed)
    {
        Payload payload = Payload(ray, seed);
        uint rayFlags = RAY_FLAG_FORCE_OPAQUE;
        for (int depth = 0; depth < 10; ++depth)
        {
            if (payload.isActive == 0)
                break;
            HitObject hit =
                HitObject::TraceRay(gScene.tlas, rayFlags, 0xff, 0, gScene.getRayKindCount(), 0, payload.ray.toRayDesc(), payload);
            ReorderThread(hit);
            HitObject::Invoke(gScene.tlas, hit, payload);
        }
        return payload.totalContribution;
    }

    float3 tracePrimary(Ray ray, uint seed)
    {
        Payload payload = Payload(ray, seed);
        uint rayFlags = RAY_FLAG_FORCE_OPAQUE;
        for (int depth = 0; depth < 10; ++depth)
        {
            if (payload.isActive == 0)
                break;
            TraceRay(gScene.tlas, rayFlags, 0xff, 0, gScene.getRayKindCount(), 0, payload.ray.toRayDesc(), payload);
        }
        return payload.totalContribution;
    }
}
