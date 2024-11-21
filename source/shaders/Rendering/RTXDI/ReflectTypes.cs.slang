/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/

/** Dummy compute program for reflection of the structured buffer types.

     Falcor requires reflection data to be able to create a structured buffer,
     but the programs are not yet ready to be used when we create the buffers.
     Therefore we create this dummy program to make it easier.
 */

#include "rtxdi/RtxdiParameters.h"

import PackedTypes;

StructuredBuffer<PackedSurfaceData> surfaceData;
StructuredBuffer<PackedPolymorphicLight> lightInfo;
StructuredBuffer<RTXDI_PackedReservoir> reservoirs;

void main() {}
