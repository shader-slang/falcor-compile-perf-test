/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import SVGFCommon;

cbuffer PerImageCB
{
    Texture2D gAlbedo;
    Texture2D gEmission;
    Texture2D gIllumination;
};

float4 main(FullScreenPassVsOut vsOut) : SV_TARGET0
{
    const int2 ipos = int2(vsOut.posH.xy);

    return gAlbedo[ipos] * gIllumination[ipos] + gEmission[ipos];
}
