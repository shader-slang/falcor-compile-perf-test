/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#pragma once
#include "Utils/HostDeviceShared.slangh"
#include "Utils/Math/MathConstants.slangh"

BEGIN_NAMESPACE_FALCOR

static const float kNRDDepthRange = 10000.0f;
static const float kNRDInvalidPathLength = HLF_MAX;
static const float kNRDMinReflectance = 0.01f;

END_NAMESPACE_FALCOR
