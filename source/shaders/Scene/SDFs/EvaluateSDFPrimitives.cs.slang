/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"

import Scene.SDFs.SDF3DPrimitive;
import Utils.Math.PackedFormats;

#define GROUP_WIDTH 4

cbuffer CB
{
    uint gGridWidth;
    uint gPrimitiveCount;
};

StructuredBuffer<SDF3DPrimitive> gPrimitives;
Texture3D<float> gOldValues;

#ifdef _USE_SD_FIELD_3D_TEXTURE
RWTexture3D<float> gValues;
#else
RWBuffer<float> gValues;
#endif

void writeDistanceToValues(uint3 valueCoords, float sd)
{
#ifdef _USE_SD_FIELD_3D_TEXTURE
    gValues[valueCoords] = sd;
#else
    uint writeLocation = flatten3D(valueCoords, gGridWidth + 1, gGridWidth + 1);
    gValues[writeLocation] = sd;
#endif
}

[numthreads(GROUP_WIDTH, GROUP_WIDTH, GROUP_WIDTH)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID)
{
    if (any(dispatchThreadID >= gGridWidth + 1)) return;

    const uint3 valueCoords = dispatchThreadID;

    // Calculate the grid position of the value coordinates.
    const float3 p = -0.5f + float3(valueCoords) / gGridWidth;

    // Evaluate the signed distance value.
    float sd = FLT_MAX;

#ifdef _MERGE_WITH_THE_SD_FIELD
    sd = gOldValues[valueCoords];
#endif

    for (uint primitiveID = 0; primitiveID < gPrimitiveCount; primitiveID++)
    {
        SDF3DPrimitive primitive = gPrimitives[primitiveID];
        sd = primitive.eval(p, sd);
    }

    writeDistanceToValues(valueCoords, sd);
}
