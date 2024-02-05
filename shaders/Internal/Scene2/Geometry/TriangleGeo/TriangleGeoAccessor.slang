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
import Internal.Scene2.Geometry.TriangleGeo.TriangleGeoSharedTypes;

import Internal.Utils.Geometry.GeometryHelpersInternal;

import Utils.NVAPI; // TODO(@skallweit) this is currently needed to make g_NvidiaExt visible
import Utils.Math.Ray;

struct TriangleHitInfo
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
    __init(TriangleHitInfo triangleHitInfo)
    {
        type = GeometryType2::Triangle;
        instanceDataIndex = triangleHitInfo.instanceDataIndex;
        geometryDataIndex = triangleHitInfo.geometryDataIndex;
        primitiveIndex = triangleHitInfo.primitiveIndex;
        barycentrics = triangleHitInfo.barycentrics;
    }
};
// clang-format on

struct TriangleGeoAccessor
{
    typedef BuiltInTriangleIntersectionAttributes Attrib;

    static uint3 getTriangleIndices(const TriangleHitInfo hit)
    {
        GeometryData geometryData = gScene.geometryData[hit.geometryDataIndex];
        uint ibOffset = geometryData.getTriangleIndexOffset() + hit.primitiveIndex * 3;
        uint vbOffset = geometryData.getTriangleVertexOffset();
        uint3 triangleIndices;
        triangleIndices.x = gScene.triangleGeoData.indices[ibOffset] + vbOffset;
        triangleIndices.y = gScene.triangleGeoData.indices[ibOffset + 1] + vbOffset;
        triangleIndices.z = gScene.triangleGeoData.indices[ibOffset + 2] + vbOffset;
        return triangleIndices;
    }

    static void getTriangleVertices(const TriangleHitInfo hit, out TriangleStaticVertex triangleVertices[3])
    {
        uint3 triangleIndices = getTriangleIndices(hit);
        triangleVertices[0] = gScene.triangleGeoData.vertices[triangleIndices[0]];
        triangleVertices[1] = gScene.triangleGeoData.vertices[triangleIndices[1]];
        triangleVertices[2] = gScene.triangleGeoData.vertices[triangleIndices[2]];
    }

    static bool getVertexData(const Ray ray, const TriangleHitInfo hit, out VertexData v)
    {
        TriangleStaticVertex vertices[3];
        getTriangleVertices(hit, vertices);

        float3 barycentrics = hit.getBarycentricWeights();

        float3 posI = vertices[0].position * barycentrics[0];
        posI += vertices[1].position * barycentrics[1];
        posI += vertices[2].position * barycentrics[2];

        float3 normalI = vertices[0].normal * barycentrics[0];
        normalI += vertices[1].normal * barycentrics[1];
        normalI += vertices[2].normal * barycentrics[2];

        float3 tangentI = vertices[0].tangent * barycentrics[0];
        tangentI += vertices[1].tangent * barycentrics[1];
        tangentI += vertices[2].tangent * barycentrics[2];

        float3 facenormalI = cross(vertices[1].position - vertices[0].position, vertices[2].position - vertices[0].position);
        if (all(normalI == float3(0.0)))
            normalI = facenormalI;

        float4x4 worldFromInstance = gScene.xformWorldFromInstance[hit.instanceDataIndex];
        float4x4 worldFromInstanceIT = transpose(gScene.xformInstanceFromWorld[hit.instanceDataIndex]);

        v.posW = mul(worldFromInstance, float4(posI, 1.f)).xyz;
        v.tangentW = float4(mul(worldFromInstance, float4(tangentI, 1.f)).xyz, 1);
        v.normalW = normalize(mul(worldFromInstanceIT, float4(normalI, 0.f)).xyz);
        v.faceNormalW = normalize(mul(worldFromInstanceIT, float4(facenormalI, 0.f)).xyz);

        v.texC = vertices[0].texCrd * barycentrics[0];
        v.texC += vertices[1].texCrd * barycentrics[1];
        v.texC += vertices[2].texCrd * barycentrics[2];

        v.curveRadius = 0.f;
        v.coneTexLODValue = 0.f;
#if 0
        {
            float3 positions[3] = { vertices[0].position, vertices[1].position, vertices[2].position };
            float2 txcoords[3] = { vertices[0].texCrd, vertices[1].texCrd, vertices[2].texCrd };
            v.coneTexLODValue = computeRayConeTriangleLODValue(positions, txcoords, float3x3(worldFromInstance));
        }
#endif
        const float3 positions[3] = { vertices[0].position, vertices[1].position, vertices[2].position };
        const float3 normals[3] = { vertices[0].normal, vertices[1].normal, vertices[2].normal };

        // TODO: This needs to be actually calculated beforehand, Scene1 computes it on the CPU,
        // but that won't work with a GPU SceneGraph
        bool isFrontFaceCW = true;
        v.normalCurvature = computeNormalCurvature(
            positions, normals, barycentrics, isFrontFaceCW, worldFromInstance, float3x3(worldFromInstanceIT), -ray.dir
        );

        return true;
    }

    static bool getVertexData(const Ray ray, const HitInfo hitInfo, out VertexData v)
    {
        v = {};
        if (hitInfo.type != getType())
            return false;
        return getVertexData(ray, TriangleHitInfo(hitInfo), v);
    }

    // Hit for reordering
    static TriangleHitInfo makeHitInfo(const HitObject hitObject)
    {
        TriangleHitInfo result;
        result.instanceDataIndex = hitObject.GetInstanceIndex();
        result.geometryDataIndex = hitObject.GetInstanceID() + hitObject.GetGeometryIndex();
        result.primitiveIndex = hitObject.GetPrimitiveIndex();
        Attrib attribs = hitObject.GetAttributes<Attrib>();
        result.barycentrics = attribs.barycentrics;
        return result;
    }

    // Hit for shader function
    static TriangleHitInfo makeHitInfo(const Attrib attribs)
    {
        TriangleHitInfo result;
        result.instanceDataIndex = InstanceIndex();
        result.geometryDataIndex = InstanceID() + GeometryIndex();
        result.primitiveIndex = PrimitiveIndex();
        result.barycentrics = attribs.barycentrics;
        return result;
    }

    // Hit for inline
    static TriangleHitInfo makeHitInfo<let Flags : int>(RayQuery<Flags> rayQuery)
    {
        TriangleHitInfo result;
        result.instanceDataIndex = rayQuery.CommittedInstanceIndex();
        result.geometryDataIndex = rayQuery.CommittedInstanceID() + rayQuery.CommittedGeometryIndex();
        result.primitiveIndex = rayQuery.CommittedPrimitiveIndex();
        result.barycentrics = rayQuery.CommittedTriangleBarycentrics();
        return result;
    }

    static GeometryType2 getType() { return GeometryType2::Triangle; }
};

struct TriangleGeoHitInfo : IHitInfo
{
    TriangleHitInfo data;
    __init() {}
    __init(TriangleHitInfo data_) { data = data_; };

    bool isValid() { return data.barycentrics[0] >= 0; }
    bool getVertexData(const Ray ray, out VertexData v) { return TriangleGeoAccessor::getVertexData(ray, data, v); }
    Ray makeScatterRay(const Ray incomingRay, const VertexData v, const float3 outDirW)
    {
        return makeScatterRayDefault(incomingRay, v, outDirW);
    }

    uint getInstanceDataIndex() { return data.instanceDataIndex; }
    uint getGeometryDataIndex() { return data.geometryDataIndex; }
    uint getPrimitiveIndex() { return data.primitiveIndex; }
};

struct TriangleGeoIntersector : IGeoIntersector
{
    uint dummy; // This dummy is here so we can dynamicObjectCast it

    typedef TriangleGeoHitInfo GeoHitInfo;

    bool getVertexData(const Ray ray, const HitInfo hitInfo, out VertexData v)
    {
        return TriangleGeoAccessor::getVertexData(ray, hitInfo, v);
    }

    bool getVertexData(const Ray ray, const GeoHitInfo hitInfo, out VertexData v)
    {
        return TriangleGeoAccessor::getVertexData(ray, hitInfo.data, v);
    }

    GeoHitInfo makeHitInfo(const HitObject hitObject)
    {
        GeoHitInfo result;
        result.data = TriangleGeoAccessor::makeHitInfo(hitObject);
        return result;
    }

    GeoHitInfo makeHitInfo(const GeoIsectAttributes geoAttr)
    {
        TriangleGeoAccessor::Attrib attr = reinterpret<TriangleGeoAccessor::Attrib, GeoIsectAttributes>(geoAttr);
        GeoHitInfo result;
        result.data = TriangleGeoAccessor::makeHitInfo(attr);
        return result;
    }

    GeoHitInfo makeHitInfo<let Flags : int>(RayQuery<Flags> rayQuery, const GeoIsectAttributes geoAttr)
    {
        TriangleGeoAccessor::Attrib attr = reinterpret<TriangleGeoAccessor::Attrib, GeoIsectAttributes>(geoAttr);
        GeoHitInfo result;
        result.data = TriangleGeoAccessor::makeHitInfo(rayQuery);
        return result;
    }

    GeometryType2 getType() { return TriangleGeoAccessor::getType(); }

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
