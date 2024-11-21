/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
cbuffer PerFrameCB
{
    float4x4 gvpTransform;
    float3 gFontColor;
};

Texture2D gFontTex;

float4 transform(float2 posS)
{
    return mul(gvpTransform, float4(posS, 0.5f, 1));
}

void vsMain(float2 posS: POSITION, inout float2 texC: TEXCOORD, out float4 posSV: SV_POSITION)
{
    posSV = transform(posS);
}

float4 calcColor(float2 texC)
{
    float4 color = gFontTex.Load(int3(texC, 0));
    color.rgb = gFontColor;
    return color;
}

float4 psMain(float2 texC: TEXCOORD) : SV_TARGET0
{
    return calcColor(texC);
}
