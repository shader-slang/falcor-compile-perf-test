/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/

/**
 * Structs and enums for GUIPass for the SDFEditor
 */

#pragma once
#include "Utils/HostDeviceShared.slangh"

#ifdef HOST_CODE
#include "Utils/Math/AABB.h"
#include "Scene/HitInfoType.slang"
#include "Scene/SDFs/SDF3DPrimitiveCommon.slang"
#else
__exported import Utils.Math.AABB;
__exported import Scene.HitInfoType;
__exported import Scene.SDFs.SDF3DPrimitive;
#endif

BEGIN_NAMESPACE_FALCOR

struct SDFGridPlane
{
    float3 position = float3(0.0f, 0.0f, 0.0f);
    float gridLineWidth = 0.04f;
    float3 normal = float3(0.0f, 0.0f, 1.0f);
    float gridScale = 25.0f;
    float3 rightVector = float3(1.0f, 0.0f, 0.0f);
    float planeSize = 0.25f;
    float4 color = float4(0.0f, 0.5f, 1.0f, 0.5f);
    uint active = 0; // Only render and manipulate the grid if this is true.
    uint3 _pad0;

    float intersect(float3 rayOrigin, float3 rayDir) // Returns the distance from the ray to the plane.
    {
        return -dot(normal, rayOrigin - position) / dot(normal, rayDir);
    }
};

enum class SDFBBRenderMode : uint32_t
{
    Disabled = 0,
    RenderAll = 1,
    RenderSelectedOnly = 2,

    Count
};

enum class SDFEditorAxis : uint32_t
{
    X = 0,
    Y = 1,
    Z = 2,
    OpSmoothing = 3,
    All = 4,

    Count
};

struct SDFBBRenderSettings
{
    uint renderMode = 2;
    uint selectedInstanceID = 0;
    float edgeThickness = 0.0001f;
};

struct SDFPickingInfo
{
    float distance;
    uint instanceID;
    HitType hitType;
};

struct SDFEditingData
{
    uint editing;
    uint previewEnabled;
    uint instanceID;
    uint scalingAxis;
    SDF3DPrimitive primitive;
    SDF3DPrimitive symmetryPrimitive;
    AABB primitiveBB;
    AABB symmetryPrimitiveBB;
};

END_NAMESPACE_FALCOR
