/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import Params;
import Utils.Math.PackedFormats;

struct ColorType
{
#if COLOR_FORMAT == 0 /* ColorFormat::RGBA32F */
    float4 data;

    float3 get()
    {
        return data.rgb;
    }

    [mutating] void set(float3 color)
    {
        data = float4(color, 1.f);
    }

#elif COLOR_FORMAT == 1 /* ColorFormat::LogLuvHDR */
    uint data;

    float3 get()
    {
        return decodeLogLuvHDR(data);
    }

    [mutating] void set(float3 color)
    {
        data = encodeLogLuvHDR(color);
    }

#else
#error Invalid COLOR_FORMAT
#endif
};
