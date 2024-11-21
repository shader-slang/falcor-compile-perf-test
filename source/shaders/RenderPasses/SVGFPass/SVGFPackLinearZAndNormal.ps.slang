/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import Utils.Math.MathHelpers;
import SVGFCommon;

cbuffer PerImageCB
{
    Texture2D gLinearZ;
    Texture2D gNormal;
};

float4 main(FullScreenPassVsOut vsOut) : SV_TARGET0
{
    float4 fragCoord = vsOut.posH;
    const int2 ipos = int2(fragCoord.xy);

    const float2 nPacked = ndir_to_oct_snorm(gNormal[ipos].xyz);
    return float4(gLinearZ[ipos].xy, nPacked.x, nPacked.y);
}
