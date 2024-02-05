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
import Internal.Scene2.VertexData;

import Utils.Math.Ray;

struct HitInfo
{
    GeometryType2 type;
    uint instanceDataIndex; // Index into scene::instanceData == CommitedInstanceIndex
    uint geometryDataIndex; // Index into scene::geometryData, NOT CommitedGeometryIndex (has offset based on specific BLAS)
    uint primitiveIndex;
    float t;
    float2 barycentrics;

    __init() { type = GeometryType2::Invalid; }
    bool isValid() { return type != GeometryType2::Invalid; }
    /// Can't do this, results in recursive import
    // bool getVertexData(const Ray ray, out VertexData v)
    // {
    //     IGeoIntersector geoIntersector = gScene.getGeoIntersector(type);
    //     return geoIntersector.makeHitInfo(q, geoIsectAttr);
    // }
};

[anyValueSize(20)]
interface IHitInfo
{
    bool isValid();
    bool getVertexData(const Ray ray, out VertexData v);
    Ray makeScatterRay(const Ray incomingRay, const VertexData v, const float3 outDirW);

    uint getInstanceDataIndex();
    uint getGeometryDataIndex();
    uint getPrimitiveIndex();
}

float3 computeRayOrigin(float3 pos, float3 normal)
{
#ifdef FALCOR_INTERNAL
    // From Carsten Waechter:
    // we move the cam to (0,0,0) and move the scene to compensate.
    //   -- otherwise the "origin" test does not really work
    // Also regarding computing the initial hitpoint, this is what we use at the moment(seems like the best variant) :
    // hitpoint.x = __fmaf_rn(u, (obj_v[1].x - obj_v[0].x), __fmaf_rn(v, (obj_v[2].x - obj_v[0].x), obj_v[0].x));
    // hitpoint.y = __fmaf_rn(u, (obj_v[1].y - obj_v[0].y), __fmaf_rn(v, (obj_v[2].y - obj_v[0].y), obj_v[0].y));
    // hitpoint.z = __fmaf_rn(u, (obj_v[1].z - obj_v[0].z), __fmaf_rn(v, (obj_v[2].z - obj_v[0].z), obj_v[0].z));
    // and: As mentioned above, due to RTX/TTU precision quirks (that we're still trying to fully understand) we also tweaked the constants
    // after the gem: 1.0/16.0 for near zero handling and the offset scale is 3x higher for the non-near zero handling. That's why we're
    // currently revisiting all of that code. It still works pretty good on RTX/TTU, but we got much more special cases over time from our
    // ISVs to "fix". So the plan is to use as much as possible from PBRT, BUT the problem is that the error handling in there NEEDS to use
    // a custom triangle intersection test, which is clearly not possible with the TTU, and as mentioned yesterday, arch is also not happy
    // to sacrifice chip space for more math/triangle.  :confused:
#endif // FALCOR_INTERNAL

    const float origin = 1.f / 16.f;
    const float fScale = 3.f / 65536.f;
    const float iScale = 3 * 256.f;

    // Per-component integer offset to bit representation of fp32 position.
    int3 iOff = int3(normal * iScale);
    float3 iPos = asfloat(asint(pos) + select(pos < 0.f, -iOff, iOff));

    // Select per-component between small fixed offset or above variable offset depending on distance to origin.
    float3 fOff = normal * fScale;
    // return select(abs(pos) < origin, pos + fOff, iPos);
    return pos + fOff;
}

Ray makeScatterRayDefault(const Ray incomingRay, const VertexData v, const float3 outDirW)
{
    const float3 rayFacingNormal = dot(incomingRay.dir, v.faceNormalW) < 0 ? v.faceNormalW : -v.faceNormalW;
    const bool reflection = dot(outDirW, rayFacingNormal) > 0;
    float3 orgW = computeRayOrigin(v.posW, reflection ? rayFacingNormal : -rayFacingNormal);
    float3 dirW = outDirW;
    return Ray(orgW, dirW);
}

struct InvalidHitInfo : IHitInfo
{
    bool isValid() { return false; }
    bool getVertexData(const Ray ray, out VertexData v)
    {
        v = {};
        return false;
    }
    Ray makeScatterRay(const Ray incomingRay, const VertexData v, const float3 outDirW) { return Ray(v.posW, outDirW); }

    uint getInstanceDataIndex() { return 0; }
    uint getGeometryDataIndex() { return 0; }
    uint getPrimitiveIndex() { return 0; }
}
