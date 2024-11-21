/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import Rendering.Volumes.IPhaseFunction;

struct VolumeProperties
{
    float3 sigmaA; ///< Absorption coefficient.. or maybe cross section.
    float3 sigmaS; ///< Scattering coefficient.. or maybe cross section.
    IPhaseFunction phaseFunction; ///< Set to NullPhaseFunction when invalid
}
