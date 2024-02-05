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
import Internal.Scene2.Geometry.GeoData;

struct RWSceneGeo
{
    // Geometry data

    /// BEGIN - PROCEDURAL GEO CODE -- to be replaced with extensions once possible
    TriangleAnimationVertexData triangleGeoData;
    CurveOTSAnimationVertexData curveOTSGeoData;
    /// END - PROCEDURAL GEO CODE

    uint sceneIntersectorsDummyValue;
    IGeoRWAccessor getGeoRWAccessor(GeometryType2 type)
    {
        return createDynamicObject<IGeoRWAccessor, uint>(int(type), sceneIntersectorsDummyValue);
    }
};

ParameterBlock<RWSceneGeo> gRWScene2Geo;
