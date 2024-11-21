/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#pragma once
#include "Utils/HostDeviceShared.slangh"

#ifdef HOST_CODE
#include "Scene/HitInfoType.slang"
#else
__exported import Scene.HitInfoType;
#endif

BEGIN_NAMESPACE_FALCOR

/**
 * Pixel data read out by the PixelInspectorPass.
 */
struct PixelData
{
    static const uint kInvalidIndex = 0xffffffff;

    // Geometry data
    float3 posW;
    float3 normal;
    float3 tangent;
    float3 bitangent;
    float3 faceNormal;
    float3 view;
    float2 texCoord;
    int frontFacing;

    // Material data
    uint materialID;
    int doubleSided;
    float opacity;
    float IoR;

    // Material instance properties
    float3 emission;
    float roughness;
    float3 guideNormal;
    float3 diffuseReflectionAlbedo;
    float3 diffuseTransmissionAlbedo;
    float3 specularReflectionAlbedo;
    float3 specularTransmissionAlbedo;
    float3 specularReflectance;
    int isTransmissive;

    // Output data
    float4 linearColor;
    float4 outputColor;
    float luminance;

    // Visibility data
    uint hitType; ///< Type of hit, see enum HitType. Only valid if instanceID != kInvalidIndex.
    uint instanceID;
    uint primitiveIndex;
    float2 barycentrics;

#ifndef HOST_CODE
    __init()
    {
        this = {};

        // V-buffer data.
        hitType = (uint)HitType::Triangle;
        instanceID = kInvalidIndex;
        primitiveIndex = kInvalidIndex;
    }
#endif
};

END_NAMESPACE_FALCOR
