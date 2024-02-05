/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
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
