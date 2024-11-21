/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
enum class NRDPathType
{
    Residual = 0,          ///< Path not assigned to any of the below types.
    Diffuse = 1,           ///< Primary hit was sampled from the diffuse BSDF lobe.
    Specular = 2,          ///< Primary hit was sampled from the specular BSDF lobe.
    DeltaReflection = 3,   ///< Primary hit was sampled as the delta reflection.
    DeltaTransmission = 4, ///< Path started with and followed delta transmission events (whenever possible - TIR could be an exception)
                           ///< until it hit the first non-delta event.
};

struct NRDRadiance
{
    uint pathType = uint(NRDPathType::Residual);
    float3 radiance = 0.f;

    NRDPathType getPathType() { return NRDPathType(pathType); }

    [mutating]
    void setPathType(NRDPathType pType) { pathType = uint(pType); }

    float3 getRadiance() { return radiance; }

    [mutating]
    void setRadiance(float3 color) { radiance = color; }
};
