/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#include "../MaterialParamLayout.slangh"

BEGIN_NAMESPACE_FALCOR

#define MATERIAL_PARAMS(PARAM) PARAM(float3, 0, baseColor, getBaseColor3, setBaseColor3, "diffuse")

DEFINE_MATERIAL_PARAM_LAYOUT(PBRTDiffuseMaterial, MATERIAL_PARAMS)

#undef MATERIAL_PARAMS

END_NAMESPACE_FALCOR
