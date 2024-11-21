/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import RenderPasses.DebugPasses.ColorMapPass.ColorMapParams;
import Utils.Color.ColorMap;

#define FORMAT_FLOAT 0
#define FORMAT_UINT 1
#define FORMAT_SINT 2

#if _FORMAT == FORMAT_FLOAT
Texture2D<float4> gTexture;
#elif _FORMAT == FORMAT_UINT
Texture2D<uint4> gTexture;
#elif _FORMAT == FORMAT_SINT
Texture2D<int4> gTexture;
#else
#error "Invalid format"
#endif

static const uint kColorMap = _COLOR_MAP;
static const uint kChannel = _CHANNEL;

cbuffer StaticCB
{
    ColorMapParams gParams;
};

float3 applyColorMap(float value)
{
    switch (kColorMap)
    {
    case ColorMap::Grey:
        return colormapGray(value);
    case ColorMap::Jet:
        return colormapJet(value);
    case ColorMap::Viridis:
        return colormapViridis(value);
    case ColorMap::Plasma:
        return colormapPlasma(value);
    case ColorMap::Magma:
        return colormapMagma(value);
    case ColorMap::Inferno:
        return colormapInferno(value);
    default:
        return 0.f;
    }
}

float4 main(float2 texC: TEXCOORD, float4 pos: SV_Position) : SV_Target0
{
    int2 pixelPos = int2(pos.xy);

    // Read raw value from selected channel.
    float value = gTexture.Load(int3(pixelPos, 0))[_CHANNEL];

    // Normalize and colorize value.
    float normalized =
        gParams.minValue == gParams.maxValue ? 0.f : saturate((value - gParams.minValue) / (gParams.maxValue - gParams.minValue));
    float3 colorized = applyColorMap(normalized);

    return float4(colorized, 1.f);
}
