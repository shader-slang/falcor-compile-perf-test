/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE(Falcor::scene2)

static const uint kRayTypeScatter = 0;
static const uint kRayTypeVisibility = 1;
static const uint kRayKindCount = 2;

static const uint kMissScatter = 0;
static const uint kMissVisibility = 1;
static const uint kMissKindCount = 2;

struct CameraData
{
    float3 position;
    float _pad0;

    float fovxInRadians;
    float nearZ;
    float farZ;
    float _pad1;

    float3 cameraU;
    float _pad2;
    float3 cameraV;
    float _pad3;
    float3 cameraW;
    float _pad4;

    float jitterX;
    float jitterY;
    float _pad5[2];
};

struct RenderUniforms
{
    uint2 frameDim;
    uint frameIndex;
    uint sampleCount = 1;
    uint2 _pad0;
};

/// BEGIN - PROCEDURAL GEO CODE
enum class GeometryType2 : uint32_t
{
    Triangle = 0,
    Sphere = 1,
    PointCloud = 2,
    CurveOTS = 3,
    Invalid
};
/// END - PROCEDURAL GEO CODE

// Per geometry data, one for each RtGeometry
struct GeometryData
{
    GeometryType2 geoType;      ///< type of the geometry
    uint geoDescIndex;          ///< index of the geometry within the geoType, matches GeometryId::index
    uint materialID = uint(-1); ///< If invalid (uint max), we might wanna use materialID from InstanceData in the future

    uint geometryIndex; ///< index of the geometry in the BLAS - CommittedGeometryIndex()

    /// Special Geometry (e.g. TriangleGeo) data for shortcircuiting going GeometryData -> MeshDesc -> ActualData
    /// Can be reused by other geometries, probably should get getter setter
    uint vbOffset;
    uint ibOffset;
};

// Per instance data, one for each TLAS instance
struct InstanceData
{
    uint globalMatrixID; ///< Points to the matrix for a given instance.
    uint instanceIndex;  ///< Index of the instance in the TLAS (not the used InstanceID) - CommitedInstanceIndex
};

struct GeoRWDesc
{
    GeometryType2 geoType;
    uint vertexIndex; ///< The index to which to write the modified data.
};

/// Vertex data that can be updated by animation handlers (e.g., skinning)
struct DynamicVertexData
{
    float3 position;
    float3 tangent;
    float3 normal;
};

/// This is a Slang 1:1 version of the RtInstanceDesc, to allow updating the instances exclusively on the GPU
struct SlangRtInstanceDesc
{
    float3x4 transform;
    uint instanceIDMask;
    uint instanceContributionFlags;
    uint2 DeviceAddress;
};

END_NAMESPACE
