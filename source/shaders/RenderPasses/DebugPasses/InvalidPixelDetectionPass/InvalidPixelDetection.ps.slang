/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
Texture2D<float4> gTexture;

float4 main(float2 texC: TEXCOORD) : SV_TARGET0
{
    uint2 uv;
    gTexture.GetDimensions(uv.x, uv.y);
    int2 xy = int2(uv * texC);
    float4 value = gTexture.Load(int3(xy, 0));
    if (any(isnan(value)))
        return float4(1, 0, 0, 1);
    else if (any(isinf(value)))
        return float4(0, 1, 0, 1);
    return float4(0, 0, 0, 1);
}
