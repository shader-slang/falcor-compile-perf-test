/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#include "../MaterialParamLayout.slangh"

BEGIN_NAMESPACE_FALCOR

static constexpr float2 kPBRTConductorMinRoughness = float2(0.05f);

#define MATERIAL_PARAMS(PARAM)                                                           \
    PARAM(float3, 0, baseColor, getBaseColor3, setBaseColor3, "eta")                     \
    PARAM(float3, 3, transmissionColor, getTransmissionColor, setTransmissionColor, "k") \
    PARAM(float2, 6, roughness, getRoughness, setRoughness, "roughness", kPBRTConductorMinRoughness)

DEFINE_MATERIAL_PARAM_LAYOUT(PBRTConductorMaterial, MATERIAL_PARAMS)

#undef MATERIAL_PARAMS

END_NAMESPACE_FALCOR
