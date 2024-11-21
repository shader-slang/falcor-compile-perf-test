/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

/** This is a host/device structure that describes a grid volume.
*/
struct GridVolumeData
{
    float4x4 transform;                             ///< Local-space to world-space transform.
    float4x4 invTransform;                          ///< World-space to local-space transform.
    float3  boundsMin               = float3(0);    ///< World-space bounds (minimum).
    float   densityScale            = 1.f;          ///< Density scale factor.
    float3  boundsMax               = float3(0);    ///< World-space bounds (maximum).
    float   emissionScale           = 1.f;          ///< Emission scale factor.
    uint    densityGrid             = 0;            ///< Index of the density grid.
    uint    emissionGrid            = 0;            ///< Index of the emission grid.
    uint    flags                   = 0;            ///< Flags (contains only emission mode for now).
    float   anisotropy              = 0.f;          ///< Phase function anisotropy (g).
    float3  albedo                  = float3(1);    ///< Medium scattering albedo.
    float   emissionTemperature     = 0.f;          ///< Emission base temperature (K).
};

END_NAMESPACE_FALCOR
