/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

/** Parameters for a DiffuseSpecularBRDF approximation.
    The fields represent a best fit diffuse/GGX approximation of a target BRDF.
*/
struct DiffuseSpecularData
{
    float3 baseColor;   ///< Base color in linear space.
    float roughness;    ///< Linearly perceptual roughness. The specular lobe uses the squared roughness.
    float specular;
    float metallic;
    float lossValue;    ///< Loss value when fitted to a target BRDF. Not used for rendering.

#ifdef HOST_CODE
    bool operator== (const DiffuseSpecularData& other) const
    {
        if (any(baseColor != other.baseColor)) return false;
        if (roughness != other.roughness) return false;
        if (specular != other.specular) return false;
        if (metallic != other.metallic) return false;
        if (lossValue != other.lossValue) return false;
        return true;
    }
    bool operator!= (const DiffuseSpecularData& other) const
    {
        return !(*this == other);
    }
#endif
};

END_NAMESPACE_FALCOR
