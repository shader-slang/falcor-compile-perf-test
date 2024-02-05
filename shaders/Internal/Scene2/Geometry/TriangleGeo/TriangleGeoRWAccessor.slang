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
import Internal.Scene2.Geometry.GeoBase;
import Internal.Scene2.RWSceneGeo;

struct TriangleGeoRWAccessor : IGeoRWAccessor
{
    uint dummy; // This dummy is here so we can dynamicObjectCast it

    DynamicVertexData getDynamicVertexData(uint vertexIndex)
    {
        DynamicVertexData result;
        result.position = gRWScene2Geo.triangleGeoData.vertices[vertexIndex].position;
        result.normal = gRWScene2Geo.triangleGeoData.vertices[vertexIndex].normal;
        result.tangent = gRWScene2Geo.triangleGeoData.vertices[vertexIndex].tangent;
        return result;
    }

    void setDynamicVertexData(uint vertexIndex, DynamicVertexData data)
    {
        gRWScene2Geo.triangleGeoData.vertices[vertexIndex].position = data.position;
        gRWScene2Geo.triangleGeoData.vertices[vertexIndex].normal = data.normal;
        gRWScene2Geo.triangleGeoData.vertices[vertexIndex].tangent = data.tangent;
    }
};
