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
import Internal.Scene2.VertexData;
import Internal.Scene2.Cameras.Camera;

import Scene.Lights.EnvMap;
import Utils.Math.Ray;

__exported import Scene.Material.MaterialSystem;

struct Scene2
{
    // Raytracing
    RaytracingAccelerationStructure tlas;

    // Geometry data

    /// BEGIN - PROCEDURAL GEO CODE -- to be replaced with extensions once possible
    TriangleGeoData triangleGeoData;
    SphereGeoData sphereGeoData;
    PointCloudGeoData pointCloudGeoData;
    CurveOTSGeoData curveOTSGeoData;
    /// END - PROCEDURAL GEO CODE

    // Instance data
    StructuredBuffer<GeometryData> geometryData;
    StructuredBuffer<InstanceData> instanceData;
    // StructuredBuffer<GeometryInstanceData> geometryInstanceDescs;
    StructuredBuffer<float4x4> xformWorldFromInstance; // addressed by instanceData[InstanceIndex].globalMatrixID
    StructuredBuffer<float4x4> xformInstanceFromWorld; // addressed by instanceData[InstanceIndex].globalMatrixID

    // Materials
    ParameterBlock<MaterialSystem> materials;

    /**
     * @brief Retrieve geometry data for a given hit
     *
     * @param instanceID - User provided instance ID, used as an offset into geometryData (where BLAS geometries start)
     * @param instanceIndex - Index of the InstanceDesc within the TLAS
     * @param geometryIndex - Index of the GeometryDesc within its BLAS
     */
    GeometryData getGeometryData(uint instanceID, uint instanceIndex, uint geometryIndex);
    {
        return geometryData[instanceID + geometryIndex];
    }

    InstanceData getInstanceData(uint instanceID, uint instanceIndex, uint geometryIndex) { return instanceData[instanceIndex]; }

    /*
    GeometryInstanceData getGeometryInstanceData(uint instanceID, uint instanceIndex, uint geometryIndex)
    {
        GeometryInstanceData result;
        result.geometry = getGeometryData(instanceID, instanceIndex, geometryIndex);
        result.instance = getInstanceData(instanceID, instanceIndex, geometryIndex);
        return result;
    }
    */

    uint sceneIntersectorsDummyValue;
    IGeoIntersector getGeoIntersector(GeometryType2 type)
    {
        return createDynamicObject<IGeoIntersector, uint>(int(type), sceneIntersectorsDummyValue);
    }

    uint rayKindCount;
    // Effectively only two options, 1 -- hit ray kind, 2 -- hit & occlusion ray kind
    uint getRayKindCount()
    {
#if !defined(FALCOR_RAY_KIND_COUNT)
        return rayKindCount;
#else
        return FALCOR_RAY_KIND_COUNT;
#endif
    }

    Camera camera;
    Camera getCamera() { return camera; }

    bool hasEnvMap;
    EnvMap envMap;
};

ParameterBlock<Scene2> gScene;
