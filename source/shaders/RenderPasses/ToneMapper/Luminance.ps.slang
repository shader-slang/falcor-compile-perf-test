/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
SamplerState gColorSampler;

Texture2D gColorTex;

float luminance(float3 color)
{
    return dot(color, float3(0.299, 0.587, 0.114));
}

float4 main(float2 texC: TEXCOORD) : SV_TARGET0
{
    float4 color = gColorTex.Sample(gColorSampler, texC);
    float logLuminance = log2(max(0.0001, luminance(color.xyz)));
    return float4(logLuminance, 0, 0, 1);
}
