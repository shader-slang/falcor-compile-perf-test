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
import Internal.Scene2.Geometry.CurveOTSGeo.CurveOTSGeoSharedTypes;

import Utils.NVAPI; // TODO(@skallweit) this is currently needed to make g_NvidiaExt visible
import Utils.Math.Ray;

struct CurveOTSHitInfo
{
    uint instanceDataIndex;
    uint geometryDataIndex;
    uint primitiveIndex;
    float2 barycentrics;

    __init() { barycentrics = float2(-1); }
    __init(HitInfo hitInfo)
    {
        instanceDataIndex = hitInfo.instanceDataIndex;
        geometryDataIndex = hitInfo.geometryDataIndex;
        primitiveIndex = hitInfo.primitiveIndex;
        barycentrics = hitInfo.barycentrics;
    }

    float3 getBarycentricWeights() { return float3(1.f - barycentrics.x - barycentrics.y, barycentrics.x, barycentrics.y); }
};

// clang-format off
extension HitInfo
{
    __init(CurveOTSHitInfo curveHitInfo)
    {
        type = GeometryType2::CurveOTS;
        instanceDataIndex = curveHitInfo.instanceDataIndex;
        geometryDataIndex = curveHitInfo.geometryDataIndex;
        primitiveIndex = curveHitInfo.primitiveIndex;
        barycentrics = curveHitInfo.barycentrics;
    }
};
// clang-format on

struct CurveOTSGeoAccessor
{
    typedef BuiltInTriangleIntersectionAttributes Attrib;

    static uint3 getTriangleIndices(const CurveOTSHitInfo hit)
    {
        GeometryData geometryData = gScene.geometryData[hit.geometryDataIndex];
        uint ibOffset = geometryData.getCurveOTSIndexOffset() + hit.primitiveIndex * 3;
        uint vbOffset = geometryData.getCurveOTSVertexOffset();
        return uint3(
                   gScene.curveOTSGeoData.indices[ibOffset],
                   gScene.curveOTSGeoData.indices[ibOffset + 1],
                   gScene.curveOTSGeoData.indices[ibOffset + 2]
               ) +
               vbOffset;
    }

    static void getTriangleVertices(const CurveOTSHitInfo hit, out CurveOTSStaticVertex triangleVertices[3])
    {
        uint3 triangleIndices = getTriangleIndices(hit);
        triangleVertices[0] = gScene.curveOTSGeoData.vertices[triangleIndices[0]];
        triangleVertices[1] = gScene.curveOTSGeoData.vertices[triangleIndices[1]];
        triangleVertices[2] = gScene.curveOTSGeoData.vertices[triangleIndices[2]];
    }

    /// TODO: Revisit range intersection
    static bool getVertexData(const Ray ray, const CurveOTSHitInfo hit, out VertexData vd)
    {
        CurveOTSStaticVertex v[3];
        getTriangleVertices(hit, v);
        const float3 barycentrics = hit.getBarycentricWeights();

        float4x4 worldFromInstance = gScene.xformWorldFromInstance[hit.instanceDataIndex];
        float4x4 worldFromInstanceIT = transpose(gScene.xformInstanceFromWorld[hit.instanceDataIndex]);

        const float3 faceNormalI = normalize(cross(v[1].position - v[0].position, v[2].position - v[0].position));
        float3 faceNormalW = mul((float3x3)worldFromInstanceIT, faceNormalI);

        const float3 posI = v[0].position * barycentrics[0] + v[1].position * barycentrics[1] + v[2].position * barycentrics[2];
        const float3 posW = mul(worldFromInstance, float4(posI, 1.f)).xyz;

        // The first vertex of each triangle is at the beginning of curve segment.
        const float3 curveVertexPosI = v[0].position - v[0].radius * v[0].fromCenter;
        const float3 curveVertexPosW = mul(worldFromInstance, float4(curveVertexPosI, 1.f)).xyz;

        const float3 viewDir = normalize(ray.origin - curveVertexPosW);
        const bool frontFacing = dot(viewDir, faceNormalW) >= 0.f;

        faceNormalW = frontFacing ? faceNormalW : -faceNormalW;
        const float3 fwd = mul((float3x3)worldFromInstance, v[0].curveTangent);
        float3 s = -cross(fwd, faceNormalW);

        const float dist = dot(s, posW - curveVertexPosW) + v[0].radius;
        float u = 1.f - (dist / (2.f * v[0].radius));

        s = normalize(-cross(fwd, viewDir));
        float3 t = normalize(-cross(s, fwd));

        float range = saturate(abs(dot(faceNormalW, t)) * 1.414f / 1.11f);
        u = saturate((u - 0.5f) * range + 0.5f);

        float cosPhi = 1.f - 2.f * u;
        float sinPhi = sqrt(max(0.f, 1.f - cosPhi * cosPhi));

        float3 nrmW = normalize(cosPhi * s + sinPhi * t);
        float3 tanW = fwd;

        vd.posW = posW;
        vd.normalW = nrmW;
        vd.faceNormalW = faceNormalW;
        vd.tangentW = float4(tanW, 1);
        vd.texC = v[0].texCrd * barycentrics[0];
        vd.texC += v[1].texCrd * barycentrics[1];
        vd.texC += v[2].texCrd * barycentrics[2];

        vd.curveRadius = v[0].radius;
        vd.coneTexLODValue = 0.f;

        return true;
    }

    static bool getVertexData(const Ray ray, const HitInfo hitInfo, out VertexData v)
    {
        v = {};
        if (hitInfo.type != getType())
            return false;
        return getVertexData(ray, CurveOTSHitInfo(hitInfo), v);
    }

    // Hit for reordering
    static CurveOTSHitInfo makeHitInfo(const HitObject hitObject)
    {
        CurveOTSHitInfo result;
        result.instanceDataIndex = hitObject.GetInstanceIndex();
        result.geometryDataIndex = hitObject.GetInstanceID() + hitObject.GetGeometryIndex();
        result.primitiveIndex = hitObject.GetPrimitiveIndex();
        Attrib attribs = hitObject.GetAttributes<Attrib>();
        result.barycentrics = attribs.barycentrics;
        return result;
    }

    // Hit for shader function
    static CurveOTSHitInfo makeHitInfo(const Attrib attribs)
    {
        CurveOTSHitInfo result;
        result.instanceDataIndex = InstanceIndex();
        result.geometryDataIndex = InstanceID() + GeometryIndex();
        result.primitiveIndex = PrimitiveIndex();
        result.barycentrics = attribs.barycentrics;
        return result;
    }

    // Hit for inline
    static CurveOTSHitInfo makeHitInfo<let Flags : int>(RayQuery<Flags> rayQuery)
    {
        CurveOTSHitInfo result;
        result.instanceDataIndex = rayQuery.CommittedInstanceIndex();
        result.geometryDataIndex = rayQuery.CommittedInstanceID() + rayQuery.CommittedGeometryIndex();
        result.primitiveIndex = rayQuery.CommittedPrimitiveIndex();
        result.barycentrics = rayQuery.CommittedTriangleBarycentrics();
        return result;
    }

    static GeometryType2 getType() { return GeometryType2::CurveOTS; }
};

struct CurveOTSGeoHitInfo : IHitInfo
{
    CurveOTSHitInfo data;
    __init() {}
    __init(CurveOTSHitInfo _data) { data = _data; };

    bool isValid() { return data.barycentrics[0] >= 0; }
    bool getVertexData(const Ray ray, out VertexData v) { return CurveOTSGeoAccessor::getVertexData(ray, data, v); }
    Ray makeScatterRay(const Ray incomingRay, const VertexData v, const float3 outDirW)
    {
        CurveOTSStaticVertex verts[3];
        CurveOTSGeoAccessor::getTriangleVertices(data, verts);

        const bool reflection = dot(outDirW, v.normalW) > 0;

        const float radius = verts[0].radius;

        const float otsWidthScale = 1.11f;
        const float3 posW = v.posW + otsWidthScale * v.normalW * radius * (reflection ? 1.f : -1.f);
        return Ray(posW, outDirW);
    }

    uint getInstanceDataIndex() { return data.instanceDataIndex; }
    uint getGeometryDataIndex() { return data.geometryDataIndex; }
    uint getPrimitiveIndex() { return data.primitiveIndex; }
};

struct CurveOTSGeoIntersector : IGeoIntersector
{
    uint dummy; // This dummy is here so we can dynamicObjectCast it

    typedef CurveOTSGeoHitInfo GeoHitInfo;

    bool getVertexData(const Ray ray, const HitInfo hitInfo, out VertexData v)
    {
        return CurveOTSGeoAccessor::getVertexData(ray, hitInfo, v);
    }

    bool getVertexData(const Ray ray, const GeoHitInfo hitInfo, out VertexData v)
    {
        return CurveOTSGeoAccessor::getVertexData(ray, hitInfo.data, v);
    }

    GeoHitInfo makeHitInfo(const HitObject hitObject)
    {
        GeoHitInfo result;
        result.data = CurveOTSGeoAccessor::makeHitInfo(hitObject);
        return result;
    }

    GeoHitInfo makeHitInfo(const GeoIsectAttributes geoAttr)
    {
        CurveOTSGeoAccessor::Attrib attr = reinterpret<CurveOTSGeoAccessor::Attrib, GeoIsectAttributes>(geoAttr);
        GeoHitInfo result;
        result.data = CurveOTSGeoAccessor::makeHitInfo(attr);
        return result;
    }

    GeoHitInfo makeHitInfo<let Flags : int>(RayQuery<Flags> rayQuery, const GeoIsectAttributes geoAttr)
    {
        CurveOTSGeoAccessor::Attrib attr = reinterpret<CurveOTSGeoAccessor::Attrib, GeoIsectAttributes>(geoAttr);
        GeoHitInfo result;
        result.data = CurveOTSGeoAccessor::makeHitInfo(rayQuery);
        return result;
    }

    GeometryType2 getType() { return CurveOTSGeoAccessor::getType(); }

    bool intersect(const Ray ray, uint geoDescOffset, out GeoIsectAttributes geoAttr, out float t)
    {
        geoAttr = {};
        t = {};
        return false;
    }
    bool intersect<let Flags : int>(RayQuery<Flags> rayQuery, out GeoIsectAttributes geoAttr, out float t)
    {
        geoAttr = {};
        t = {};
        return false;
    }
};
