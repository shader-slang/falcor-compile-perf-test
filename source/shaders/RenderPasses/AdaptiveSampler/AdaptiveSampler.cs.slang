/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/

/** Compute shader implementing an adaptive sampling distribution pass based on an importance map.

     The importance map is an input to the pass and the pass generates an R8Uint buffer containing
     the number of samples to use for each pixel

     The dispatch dimension is over pixels (XY).
 */

import AdaptiveSamplerTypes;

cbuffer PerFrameCB
{
    uint2 gResolution;
    float gMinimumSPP;
    float gAverageSPP;
    uint gMaximumSPP;
    float gPrecomputedFactor;
    uint gTextureIndex;
    uint gBNTextureWidth;
    uint gBNTextureHeight;
    float gShowErrorMin;
    float gShowErrorMax;
}

// Input and output data to compute the distribution of sample.
static const uint kNumberOfBNTextures = 64;
Texture2D<float> gBlueNoiseTextures2D[kNumberOfBNTextures];
Texture2D<float> gBlueNoiseTextures2Dx1D[kNumberOfBNTextures];
Texture2D<float> gImportanceMap;
RWTexture2D<uint> gOutput;
RWTexture2D<float4> gUpperLimitClampingMap;

static const AdaptiveSamplerMode kMode = AdaptiveSamplerMode(ADAPTIVE_SAMPLER_MODE);
static const AdaptiveSamplerClampMode kClampMode = AdaptiveSamplerClampMode(CLAMP_MODE);
static const bool kUsePowersOfTwo = bool(USE_POWERS_OF_TWO);
static const AdaptiveSamplerAnimMode kAnimationMode = AdaptiveSamplerAnimMode(ADAPTIVE_SAMPLER_ANIMATION_MODE);
static const bool kShowError = bool(SHOW_ERROR);

static const float kGoldenRatioConjugate = 0.61803398875f;

// Internal buffers.
RWBuffer<float> gImportanceSum;
RWBuffer<float4> gUpperLimitClampingSums;

[numthreads(16, 16, 1)]
void createSampleDistributionMap(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (any(dispatchThreadId.xy >= gResolution))
        return;
    const uint2 pixelPos = dispatchThreadId.xy;

    // Calculate the "random" value, called rnd, for the current pixel.
    float rnd = 0.5f; // Thresholding is the default, which just rounds to nearest integer, i.e.., rnd = 0.5 here.
    if (kMode != AdaptiveSamplerMode::Thresholding)
    {
        // Find pseudo-random numbers in [0,1] to make it randomly round up or down.
        if (kMode == AdaptiveSamplerMode::BlueNoiseDithered2D)
        {
            uint index = (kAnimationMode == AdaptiveSamplerAnimMode::Animated) ? gTextureIndex : 0;
            rnd = gBlueNoiseTextures2D[index].Load(int3(pixelPos.x % gBNTextureWidth, pixelPos.y % gBNTextureHeight, 0)).x;
        }
        else if (kMode == AdaptiveSamplerMode::BlueNoiseDithered2Dx1D)
        {
            uint index = (kAnimationMode == AdaptiveSamplerAnimMode::Animated) ? gTextureIndex : 0;
            rnd = gBlueNoiseTextures2Dx1D[index].Load(int3(pixelPos.x % gBNTextureWidth, pixelPos.y % gBNTextureHeight, 0)).x;
        }

        // Apply golden ratio animation.
        if (kAnimationMode == AdaptiveSamplerAnimMode::GoldenRatioAnimated)
        {
            rnd = frac(rnd + kGoldenRatioConjugate * float(gTextureIndex));
        }
    }

    // Use the rnd value to calculate the number of samples that should be used for this pixel.
    uint numSamples;
    float numSamplesFloat;
    {
        float factor = gPrecomputedFactor / gImportanceSum[0];

        if (kClampMode == AdaptiveSamplerClampMode::Standard) // Do no extra work for the budget that gets clamped.
        {
            numSamplesFloat = gImportanceMap[pixelPos] * factor + gMinimumSPP;
        }
        else // Use one of the methods for exploiting the sample budget that gets clamped.
        {
            if (gUpperLimitClampingMap[pixelPos].x == 0.0f) // Pixel was clamped, so set it to max.
            {
                numSamplesFloat = gMaximumSPP;
            }
            else // Pixel was not clamped, so make sure we use the excess budget (i.e., the budget that has been clamped).
            {
                float numNonClampedPixels = gUpperLimitClampingSums[0].x;
                float sumImportanceNonClampledPixels = gUpperLimitClampingSums[0].y;
                float excessTotal = gUpperLimitClampingSums[0].z;

                if (kClampMode == AdaptiveSamplerClampMode::Addition)
                {
                    numSamplesFloat = (gImportanceMap[pixelPos] + excessTotal / numNonClampedPixels) * factor + gMinimumSPP;
                }
                else // AdaptiveSamplerClampMode::Multiplication.
                {
                    float amplificationFactor = (excessTotal + sumImportanceNonClampledPixels) / numNonClampedPixels;
                    float updatedAverageSPP = factor * amplificationFactor + gMinimumSPP;
                    numSamplesFloat = gImportanceMap[pixelPos] * (updatedAverageSPP - gMinimumSPP) * numNonClampedPixels /
                                          sumImportanceNonClampledPixels +
                                      gMinimumSPP;
                }
            }
        }

        if (kUsePowersOfTwo == false)
        {
            numSamples = uint(numSamplesFloat + rnd);
        }
        else // Output only 2^k samples per pixel.
        {
            uint s = uint(numSamplesFloat);
            uint base, delta;
            if (s == 0)
            {
                base = 0;
                delta = 1;
            }
            else
            {
                base = 1 << firstbithigh(s);
                delta = base;
            }
            float fraction = (numSamplesFloat - base) / delta;
            numSamples = base + (rnd < fraction ? delta : 0);
        }
    }

    if (kShowError)
    {
        float error = numSamplesFloat - float(numSamples);
        float value = (error - gShowErrorMin) / (gShowErrorMax - gShowErrorMin);
        gOutput[pixelPos] = uint(value * 255.0f);
        return;
    }

    gOutput[pixelPos] = min(numSamples, gMaximumSPP);
}

[numthreads(16, 16, 1)]
void createUpperLimitClampingMap(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (any(dispatchThreadId.xy >= gResolution))
        return;
    const uint2 pixelPos = dispatchThreadId.xy;

    const float upperLimitClamp = (gMaximumSPP - gMinimumSPP) * gImportanceSum[0] / gPrecomputedFactor;
    const bool pixelIsClamped = gImportanceMap[pixelPos] > upperLimitClamp;
    gUpperLimitClampingMap[pixelPos].x = !pixelIsClamped ? 1.0f : 0.0f; // Store 1 for non-clamped pixels, otherwise 0.0. This is so we can
                                                                        // compute the number of non-clamped pixels.
    gUpperLimitClampingMap[pixelPos].y = !pixelIsClamped ? gImportanceMap[pixelPos] : 0.0f; // Store importance for non-clamped pixels,
                                                                                            // otherwise 0.0.
    gUpperLimitClampingMap[pixelPos].z = pixelIsClamped ? gImportanceMap[pixelPos] - upperLimitClamp : 0.0f; // Excess.
    gUpperLimitClampingMap[pixelPos].w = 0.0f;                                                               // Not used at the moment.
}
