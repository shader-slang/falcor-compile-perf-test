/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
struct VSIn
{
    float2 pos : POSITION;
};

struct VSOut
{
    float4 pos : SV_POSITION;
};

VSOut vsMain(VSIn vIn)
{
    VSOut vOut;
    vOut.pos = float4(vIn.pos, 0.f, 1.f);
    return vOut;
}

float4 psMain(VSOut vsOut, uint triangleIndex: SV_PrimitiveID) : SV_TARGET
{
    return float4(0.5f);
}
