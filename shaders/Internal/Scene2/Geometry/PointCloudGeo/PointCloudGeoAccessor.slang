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
import Internal.Scene2.Scene2;
import Internal.Scene2.HitInfo;
import Internal.Scene2.Geometry.GeoBase;
import Internal.Scene2.VertexData;
import Internal.Scene2.Geometry.PointCloudGeo.PointCloudGeoSharedTypes;

import Utils.NVAPI; // TODO(@skallweit) this is currently needed to make g_NvidiaExt visible
import Utils.Math.Ray;
import Utils.Geometry.IntersectionHelpers;

struct PointCloudHitInfo
{
    uint instanceDataIndex;
    uint geometryDataIndex;
    uint primitiveIndex;
    float t;

    __init() { t = -1; }
    __init(HitInfo hitInfo)
    {
        instanceDataIndex = hitInfo.instanceDataIndex;
        geometryDataIndex = hitInfo.geometryDataIndex;
        primitiveIndex = hitInfo.primitiveIndex;
        t = hitInfo.t;
    }
};

// clang-format off
extension HitInfo
{
    __init(PointCloudHitInfo sphereHitInfo)
    {
        type = GeometryType2::PointCloud;
        instanceDataIndex = sphereHitInfo.instanceDataIndex;
        geometryDataIndex = sphereHitInfo.geometryDataIndex;
        primitiveIndex = sphereHitInfo.primitiveIndex;
        t = sphereHitInfo.t;
    }
};
// clang-format on

struct PointCloudGeoAccessor
{
    struct Attrib
    {
        int dummy;
    };

    static bool getVertexData(const Ray ray, const PointCloudHitInfo hit, out VertexData v)
    {
        GeometryData gd = gScene.geometryData[hit.geometryDataIndex];
        PointDesc pointDesc = gScene.pointCloudGeoData.points[gd.getPointCloudPointOffset() + hit.primitiveIndex];
        InstanceData instanceDesc = gScene.instanceData[hit.instanceDataIndex];
        const float4x4 worldFromInstance = gScene.xformWorldFromInstance[instanceDesc.globalMatrixID];

        v.posW = ray.origin + ray.dir * hit.t;
        float3 centerW = mul(worldFromInstance, float4(pointDesc.center, 1.f)).xyz;
        v.normalW = normalize(v.posW - centerW);
        v.faceNormalW = v.normalW;

        v.texC = { 0.f };
        v.tangentW = { 0.f };
        v.curveRadius = 0.f;
        v.coneTexLODValue = 0.f;

        return true;
    }

    static bool getVertexData(const Ray ray, const HitInfo hitInfo, out VertexData v)
    {
        v = {};
        if (hitInfo.type != getType())
            return false;
        return getVertexData(ray, PointCloudHitInfo(hitInfo), v);
    }

    // Hit for reordering
    static PointCloudHitInfo makeHitInfo(const HitObject hitObject)
    {
        PointCloudHitInfo result;
        result.instanceDataIndex = hitObject.GetInstanceIndex();
        result.geometryDataIndex = hitObject.GetInstanceID() + hitObject.GetGeometryIndex();
        result.primitiveIndex = hitObject.GetPrimitiveIndex();
        result.t = hitObject.GetRayDesc().TMax;
        return result;
    }

    // Hit for shader function
    static PointCloudHitInfo makeHitInfo(const Attrib)
    {
        PointCloudHitInfo result;
        result.instanceDataIndex = InstanceIndex();
        result.geometryDataIndex = InstanceID() + GeometryIndex();
        result.primitiveIndex = PrimitiveIndex();
        result.t = RayTCurrent();
        return result;
    }

    // Hit for inline
    static PointCloudHitInfo makeHitInfo<let Flags : int>(RayQuery<Flags> rayQuery, const Attrib)
    {
        PointCloudHitInfo result;
        result.instanceDataIndex = rayQuery.CommittedInstanceIndex();
        result.geometryDataIndex = rayQuery.CommittedInstanceID() + rayQuery.CommittedGeometryIndex();
        result.primitiveIndex = rayQuery.CommittedPrimitiveIndex();
        result.t = rayQuery.CommittedRayT();
        return result;
    }

    static GeometryType2 getType() { return GeometryType2::PointCloud; }

    static bool intersect(const Ray ray, uint geoDescOffset, out Attrib attr, out float t)
    {
        attr = {};
        t = {};
        const PointDesc desc = gScene.pointCloudGeoData.points[geoDescOffset];
        return intersectRaySphere(ray.origin, ray.dir, desc.center, desc.radius, t, ray.tMin, ray.tMax, true);
    }

    static bool intersect(const Ray ray, uint instanceID, uint geometryIndex, uint primitiveIndex, out Attrib attr, out float t)
    {
        const GeometryData gd = gScene.geometryData[instanceID + geometryIndex];
        uint geoDescOffset = gd.getPointCloudPointOffset() + primitiveIndex;
        return intersect(ray, geoDescOffset, attr, t);
    }

    static bool intersect<let Flags : int>(RayQuery<Flags> rayQuery, out Attrib attr, out float t)
    {
        Ray ray;
        ray.origin = rayQuery.CandidateObjectRayOrigin();
        ray.dir = rayQuery.CandidateObjectRayDirection();
        ray.tMin = rayQuery.RayTMin();
        ray.tMax = rayQuery.CommittedRayT();

        return intersect(
            ray, rayQuery.CandidateInstanceID(), rayQuery.CandidateGeometryIndex(), rayQuery.CandidatePrimitiveIndex(), attr, t
        );
    }

    /// --- helpers for the handling the GeoIsectAttributes instead of Attrib

    static PointCloudHitInfo makeHitInfo(const GeoIsectAttributes attr)
    {
        Attrib localAttr;
        localAttr = reinterpret<Attrib, GeoIsectAttributes>(attr);
        return makeHitInfo(localAttr);
    }

    static PointCloudHitInfo makeHitInfo<let Flags : int>(RayQuery<Flags> rayQuery, const GeoIsectAttributes attr)
    {
        Attrib localAttr;
        localAttr = reinterpret<Attrib, GeoIsectAttributes>(attr);
        return makeHitInfo(rayQuery, localAttr);
    }

    static bool intersect(const Ray ray, uint geoDescIndex, out GeoIsectAttributes attr, out float t)
    {
        Attrib localAttr;
        bool ok = intersect(ray, geoDescIndex, localAttr, t);
        attr = reinterpret<GeoIsectAttributes, Attrib>(localAttr);
        return ok;
    }

    static bool intersect(const Ray ray, uint instanceID, uint geometryIndex, uint primitiveIndex, out GeoIsectAttributes attr, out float t)
    {
        Attrib localAttr;
        bool ok = intersect(ray, instanceID, geometryIndex, primitiveIndex, localAttr, t);
        attr = reinterpret<GeoIsectAttributes, Attrib>(localAttr);
        return ok;
    }

    static bool intersect<let Flags : int>(RayQuery<Flags> rayQuery, out GeoIsectAttributes attr, out float t)
    {
        Attrib localAttr;
        bool ok = intersect(rayQuery, localAttr, t);
        attr = reinterpret<GeoIsectAttributes, Attrib>(localAttr);
        return ok;
    }
};

struct PointCloudGeoHitInfo : IHitInfo
{
    PointCloudHitInfo data;
    __init() {}
    __init(PointCloudHitInfo _data) { data = _data; };

    bool isValid() { return data.t >= 0; }
    bool getVertexData(const Ray ray, out VertexData v) { return PointCloudGeoAccessor::getVertexData(ray, data, v); }
    Ray makeScatterRay(const Ray incomingRay, const VertexData v, const float3 outDirW)
    {
        Ray result = makeScatterRayDefault(incomingRay, v, outDirW);
        result.tMin = 1e-3f;
        return result;
    }

    uint getInstanceDataIndex() { return data.instanceDataIndex; }
    uint getGeometryDataIndex() { return data.geometryDataIndex; }
    uint getPrimitiveIndex() { return data.primitiveIndex; }
};

struct PointCloudGeoIntersector : IGeoIntersector
{
    uint dummy; // This dummy is here so we can dynamicObjectCast it

    typedef PointCloudGeoHitInfo GeoHitInfo;

    bool getVertexData(const Ray ray, const HitInfo hitInfo, out VertexData v)
    {
        return PointCloudGeoAccessor::getVertexData(ray, hitInfo, v);
    }

    bool getVertexData(const Ray ray, const GeoHitInfo hitInfo, out VertexData v)
    {
        return PointCloudGeoAccessor::getVertexData(ray, hitInfo.data, v);
    }

    GeoHitInfo makeHitInfo(const HitObject hitObject)
    {
        GeoHitInfo result;
        result.data = PointCloudGeoAccessor::makeHitInfo(hitObject);
        return result;
    }

    GeoHitInfo makeHitInfo(const GeoIsectAttributes attr)
    {
        GeoHitInfo result;
        result.data = PointCloudGeoAccessor::makeHitInfo(attr);
        return result;
    }

    GeoHitInfo makeHitInfo<let Flags : int>(RayQuery<Flags> rayQuery, const GeoIsectAttributes attr)
    {
        GeoHitInfo result;
        result.data = PointCloudGeoAccessor::makeHitInfo(rayQuery, attr);
        return result;
    }

    GeometryType2 getType() { return PointCloudGeoAccessor::getType(); }

    bool intersect(const Ray ray, uint geoDescOffset, out GeoIsectAttributes attr, out float t)
    {
        return PointCloudGeoAccessor::intersect(ray, geoDescOffset, attr, t);
    }

    bool intersect<let Flags : int>(RayQuery<Flags> rayQuery, out GeoIsectAttributes attr, out float t)
    {
        return PointCloudGeoAccessor::intersect<Flags>(rayQuery, attr, t);
    }
};
