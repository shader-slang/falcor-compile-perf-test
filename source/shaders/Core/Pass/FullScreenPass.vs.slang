/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
struct VsOut
{
    float2 texC : TEXCOORD;
#ifndef _VIEWPORT_MASK
    float4 posH : SV_POSITION;
#else
    float4 posH : POSITION;
#endif
};

VsOut main(float4 posS: POSITION, float2 texC: TEXCOORD)
{
    VsOut vOut;
    vOut.texC = texC;
    vOut.posH = posS;
    return vOut;
}
