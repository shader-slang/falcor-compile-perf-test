/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/

/** Compute shader implementing temporally weighted variance and average.

    The use case is to compute a variance map and feed it to adaptive sampling.

     The dispatch dimension is over pixels (XY).
 */

import Utils.Color.ColorHelpers;

cbuffer PerFrameCB
{
    uint2 gResolution;
    float gTemporalWeight;
    bool gResetAccumulation;
}

Texture2D<float4> gInput;
Texture2D<float2> gMotionVectors;
Texture2D<float2> gVarianceAndAverageTexture;
RWTexture2D<float2> gAccumulatedVarianceAndAverage;

static const bool kUseMotionVectors = bool(USE_MOTION_VECTORS);

SamplerState gBilinearSampler;

[numthreads(16, 16, 1)]
void createVarianceMap(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (any(dispatchThreadId.xy >= gResolution))
        return;
    const uint2 pixelPos = dispatchThreadId.xy;

    float3 Lab = linearRGBToCIELab(gInput[pixelPos].rgb);
    float luminance = Lab.r;

    // Exponentially weighted moving variance and standard deviation:
    // https://en.wikipedia.org/wiki/Moving_average#Exponentially_weighted_moving_variance_and_standard_deviation
    float prevVariance, prevAverage, average, variance;
    if (kUseMotionVectors)
    {
        float2 prevPos = float2(pixelPos) + float2(0.5f) + gMotionVectors[pixelPos].xy * gResolution;
        float2 txcoords = prevPos / gResolution;
        if (txcoords.x < 0.0f || txcoords.x > 1.0f || txcoords.y < 0.0f || txcoords.y > 1.0f)
        {
            prevAverage = luminance;
            prevVariance = 0.0f;
        }
        else
        {
            float2 varianceAndAverage = gVarianceAndAverageTexture.SampleLevel(gBilinearSampler, txcoords, 0).xy;
            prevVariance = varianceAndAverage.x;
            prevAverage = varianceAndAverage.y;
        }
    }
    else
    {
        prevVariance = gAccumulatedVarianceAndAverage[pixelPos].x;
        prevAverage = gAccumulatedVarianceAndAverage[pixelPos].y;
    }
    if (gResetAccumulation)
    {
        average = luminance;
        variance = 0.0f;
    }
    else
    {
        float delta = luminance - prevAverage;
        average = prevAverage + gTemporalWeight * delta; // gTemporalWeight * luminance + (1.0f - gTemporalWeight) * prevAverage
        variance = (1.0f - gTemporalWeight) * (prevVariance + gTemporalWeight * delta * delta);
    }

    gAccumulatedVarianceAndAverage[pixelPos] = float2(variance, average);
}
