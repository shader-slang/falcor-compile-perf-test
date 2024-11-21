/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#include "MaterialParamLayout.slangh"

BEGIN_NAMESPACE_FALCOR

#define MATERIAL_PARAMS(PARAM)                                                                                             \
    PARAM(float3, 0, baseColor, getBaseColor3, setBaseColor3, "base_color")                                                \
    PARAM(float, 3, metallic, getMetallic, setMetallic, "metallic")                                                        \
    PARAM(float, 4, roughness, getRoughness, setRoughness, "roughness", 0.05f)                                             \
    PARAM(float, 5, ior, getIndexOfRefraction, setIndexOfRefraction, "ior", 1e-3f, 1e3f)                                   \
    PARAM(float3, 6, transmissionColor, getTransmissionColor, setTransmissionColor, "transmission_color")                  \
    PARAM(float, 9, diffuseTransmission, getDiffuseTransmission, setDiffuseTransmission, "diffuse_transmission", 0.f)      \
    PARAM(float, 10, specularTransmission, getSpecularTransmission, setSpecularTransmission, "specular_transmission", 0.f) \
    PARAM(float3, 11, emissiveColor, getEmissiveColor, setEmissiveColor, "emissive_color")                                 \
    PARAM(float, 14, emissiveFactor, getEmissiveFactor, setEmissiveFactor, "emissive_factor", 1e-8f, 1e8f)

DEFINE_MATERIAL_PARAM_LAYOUT(StandardMaterial, MATERIAL_PARAMS)

#undef MATERIAL_PARAMS

END_NAMESPACE_FALCOR
