/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#pragma once

#include "Utils/HostDeviceShared.slangh"

#ifdef HOST_CODE
#include "Utils/SDF/SDFOperationType.slang"
#else
__exported import Utils.SDF.SDFOperationType;
#endif

BEGIN_NAMESPACE_FALCOR

enum class SDF3DShapeType
{
    Sphere,         ///< A sphere, defined by a radius (1 float).
    Ellipsoid,      ///< An ellipsoid, defined by three axis aligned radii (3 floats).
    Box,            ///< A box, defined by three axis aligned half extents (3 floats).
    Torus,          ///< A torus, defined by a radius (1 floats). Note that this type requires primitive blobbing to have thickness.
    Cone,           ///< A cone defined by tan of the cone angle and a height (1 floats).
    Capsule,        ///< A capsule, defined by a half length (1 float). Note that this type requires primitive blobbing to have thickness.

    Count
};

struct SDF3DPrimitive
{
    SDF3DShapeType shapeType;           ///< The shape type.
    float3 shapeData;                   ///< The shape data used to create the shape.
    float shapeBlobbing;                ///< Blobbing that should be applied to the shape.
    SDFOperationType operationType;     ///< The operation type.
    float operationSmoothing;           ///< Smoothing that should be applied to the operation.
    float3 translation;                 ///< Translation.
    // float3x3 here is okay, as this is not used in CB, only structured buffers, so no padding issues.
    float3x3 invRotationScale;          ///< Inverted rotation and scale matrix.
};

END_NAMESPACE_FALCOR
