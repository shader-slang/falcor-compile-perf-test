/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
SamplerState gSampler;
Texture2D gFont;
Texture2D guiImage;

cbuffer PerFrameCB
{
    float2 scale;
    float2 offset;
    bool useGuiImage;
};

struct VsIn
{
    float2 pos : POSITION;
    float2 texC : TEXCOORD0;
    float4 color : COLOR;
};

struct VsOut
{
    float4 color : COLOR;
    float2 texC : TEXCOORD0;
    float4 pos : SV_POSITION;
};

VsOut vsMain(VsIn vIn)
{
    VsOut vOut;
    vOut.color = vIn.color;
    vOut.texC = vIn.texC;
    vOut.pos.xy = vIn.pos.xy * scale + offset;
    vOut.pos.zw = float2(0, 1);
    return vOut;
}

float4 psMain(VsOut vOut) : SV_TARGET
{
    float fontMask = gFont.Sample(gSampler, vOut.texC).r;
    float4 c = (useGuiImage ? float4(guiImage.Sample(gSampler, vOut.texC).xyzw) : vOut.color * fontMask);
    return c;
}
