/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"
#include "flip.hlsli"

import ToneMappers;
import Utils.Color.ColorHelpers;

Texture2D gTestImage;
Texture2D gReferenceImage;
RWTexture2D<float4> gFLIPErrorMap;
RWTexture2D<float4> gFLIPErrorMapDisplay;
RWTexture2D<float4> gExposureMapDisplay;

cbuffer PerFrameCB
{
    uint2 gResolution;
    bool gIsHDR;
    bool gUseMagma;
    bool gClampInput;

    // HDR-FLIP constants.
    float gStartExposure;
    float gExposureDelta;
    uint gNumExposures;

    //  Viewing conditions for PPD calculation.
    uint gMonitorWidthPixels;
    float gMonitorWidthMeters;
    float gMonitorDistance;
};

static const FLIPToneMapperType kHDRFLIPToneMapper = FLIPToneMapperType(TONE_MAPPER);

static const float gqc = 0.7f;
static const float gpc = 0.4f;
static const float gpt = 0.95f;
static const float gw = 0.082f;
static const float gqf = 0.5f;

static float MaxDistance =
    pow(HyAB(Hunt(linearRGBToCIELab(float3(0.0f, 1.0f, 0.0f))), Hunt(linearRGBToCIELab(float3(0.0f, 0.0f, 1.0f)))), gqc);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//  Pixels per degree (PPD)
float ppd(void)
{
    return gMonitorDistance * (gMonitorWidthPixels / gMonitorWidthMeters) * (M_PI / 180.0f);
}

float3 getPixel(Texture2D image, int2 pixel, float exposure)
{
    float3 pixelCol = image[pixel].rgb;
    if (gIsHDR)
    {
        pixelCol = gClampInput ? max(pixelCol, 0.0f) : pixelCol;
        pixelCol = pow(2.0f, exposure) * pixelCol; // Exposure compensation.
        pixelCol = toneMap(pixelCol, kHDRFLIPToneMapper);
    }
    else
    {
        pixelCol = gClampInput ? clamp(pixelCol, 0.0f, 1.0f) : pixelCol;
    }

    return linearRGBToYCxCz(pixelCol);
}

float calculateWeight(float dist2, float4 ab)
{
    float b1Inv = 1.0f / ab.z;
    float b2Inv = 1.0f / ab.w;
    return ab.x * sqrt(M_PI * b1Inv) * exp(b1Inv * dist2) + ab.y * sqrt(M_PI * b2Inv) * exp(b2Inv * dist2);
}

float HyAB(float3 a, float3 b)
{
    float3 diff = a - b;

    return abs(diff.x) + length(diff.yz);
}

float3 Hunt(float3 color)
{
    float huntValue = 0.01f * color.x;
    return float3(color.x, huntValue * color.y, huntValue * color.z);
}

float redistributeErrors(float colorDifference, float featureDifference)
{
    float error = pow(colorDifference, gqc);

    //  Normalization.
    float perceptualCutoff = gpc * MaxDistance;

    if (error < perceptualCutoff)
    {
        error *= (gpt / perceptualCutoff);
    }
    else
    {
        error = gpt + ((error - perceptualCutoff) / (MaxDistance - perceptualCutoff)) * (1.0f - gpt);
    }

    error = pow(error, (1.0f - featureDifference));

    return error;
}

float LDRFLIP(uint2 pixel, float exposure = 0.0f)
{
    // ************* SETUP ************** //
    // Compute pixels per degree.
    const float pixelsPerDegree = ppd();
    const float dx = 1.0f / ppd();

    // Variables for CSF filtering.
    const float4 abValuesA = { 1.0f, 0.0f, 0.0047f, 1.0e-5f };  //  a1, a2, b1, b2 for A.
    const float4 abValuesRG = { 1.0f, 0.0f, 0.0053f, 1.0e-5f }; //  a1, a2, b1, b2 for RG.
    const float4 abValuesBY = { 34.1f, 13.5f, 0.04f, 0.025f };  //  a1, a2, b1, b2 for BY.
    float3 colorWeight = { 0.0f, 0.0f };
    float3 csfKernelSum = float3(0.0f, 0.0f, 0.0f);
    float3 referenceColorSum = float3(0.0f, 0.0f, 0.0f);
    float3 testColorSum = float3(0.0f, 0.0f, 0.0f);
    float3 spatialFilteredReference = float3(0.0f);
    float3 spatialFilteredTest = float3(0.0f);

    // Variables for feature detection.
    float sigmaFeatures = 0.5f * gw * pixelsPerDegree;
    float sigmaFeaturesSquared = sigmaFeatures * sigmaFeatures;
    float positiveKernelSum = 0.0f;
    float negativeKernelSum = 0.0f;
    float edgeKernelSum = 0.0f;
    float2 pointWeight = { 0.0f, 0.0f };
    float2 edgeWeight = { 0.0f, 0.0f };
    float2 pointNormalization = { 0.0f, 0.0f };
    float2 edgeNormalization = { 0.0f, 0.0f };
    float2 referenceEdgeGradient = float2(0.0f, 0.0f);
    float2 referencePointGradient = float2(0.0f, 0.0f);
    float2 testEdgeGradient = float2(0.0f, 0.0f);
    float2 testPointGradient = float2(0.0f, 0.0f);

    // Use radius of the spatial filter kernel, as it is always greater than or equal to the radius of the feature detection kernel.
    // See FLIP paper for explanation of the 0.04 and 3.0 factors.
    int radius = int(ceil(3.0f * sqrt(0.04f / (2.0f * M_PI * M_PI)) * pixelsPerDegree));

    // Prepare point and edge kernel sums for feature detection.
    for (int y = -radius; y <= radius; y++)
    {
        for (int x = -radius; x <= radius; x++)
        {
            int2 pixelNeighbor = pixel + int2(x, y);
            pixelNeighbor = min(max(int2(0, 0), pixelNeighbor), gResolution - 1);

            float g = exp(-(x * x + y * y) / (2.0f * sigmaFeaturesSquared));

            pointWeight = (float2(x * x, y * y) / sigmaFeaturesSquared - 1) * g;
            positiveKernelSum += (pointWeight.x >= 0.0f ? pointWeight.x : 0.0f);
            negativeKernelSum += (pointWeight.x < 0.0f ? -pointWeight.x : 0.0f);

            edgeWeight = -float2(x, y) * g;
            edgeKernelSum += (edgeWeight.x >= 0.0f ? edgeWeight.x : 0.0f);
        }
    }

    //[unroll]
    for (int y = -radius; y <= radius; y++)
    {
        //[unroll]
        for (int x = -radius; x <= radius; x++)
        {
            // Load pixel colors.
            int2 pixelNeighbor = pixel + int2(x, y);
            pixelNeighbor = min(max(int2(0, 0), pixelNeighbor), gResolution - 1);
            float3 referenceColor = getPixel(gReferenceImage, pixelNeighbor, exposure);
            float3 testColor = getPixel(gTestImage, pixelNeighbor, exposure);

            // **************** COLOR PIPELINE ******************** //
            float2 p = float2(x, y) * dx;
            float dist2 = -(p.x * p.x + p.y * p.y) * M_PI * M_PI;
            float weightA = calculateWeight(dist2, abValuesA);
            float weightRG = calculateWeight(dist2, abValuesRG);
            float weightBY = calculateWeight(dist2, abValuesBY);
            colorWeight = float3(weightA, weightRG, weightBY);

            csfKernelSum += colorWeight;
            referenceColorSum += colorWeight * referenceColor;
            testColorSum += colorWeight * testColor;

            // **************** FEATURE PIPELINE ******************** //
            float g = exp(-(x * x + y * y) / (2.0f * sigmaFeaturesSquared));

            pointWeight = (float2(x * x, y * y) / sigmaFeaturesSquared - 1) * g;
            pointNormalization = float2(1.0f, 1.0f) / float2(
                                                          pointWeight.x >= 0.0f ? positiveKernelSum : negativeKernelSum,
                                                          pointWeight.y >= 0.0f ? positiveKernelSum : negativeKernelSum
                                                      );
            edgeWeight = -float2(x, y) * g;
            edgeNormalization = float2(1.0f, 1.0f) / float2(edgeKernelSum, edgeKernelSum);

            float referenceLuminance = (referenceColor.x + 16.0f) / 116.0f; // Normalized Y from YCxCz.
            referencePointGradient += referenceLuminance * pointWeight * pointNormalization;
            referenceEdgeGradient += referenceLuminance * edgeWeight * edgeNormalization;

            float testLuminance = (testColor.x + 16.0f) / 116.0f; // Normalized Y from YCxCz.
            testPointGradient += testLuminance * pointWeight * pointNormalization;
            testEdgeGradient += testLuminance * edgeWeight * edgeNormalization;
        }
    }

    // **************** COLOR PIPELINE ******************** //
    spatialFilteredReference = referenceColorSum / csfKernelSum;
    spatialFilteredTest = testColorSum / csfKernelSum;
    spatialFilteredReference = clamp(YCxCzToLinearRGB(spatialFilteredReference), 0.0f, 1.0f);
    spatialFilteredTest = clamp(YCxCzToLinearRGB(spatialFilteredTest), 0.0f, 1.0f);
    float colorDiff = HyAB(Hunt(linearRGBToCIELab(spatialFilteredReference)), Hunt(linearRGBToCIELab(spatialFilteredTest)));

    // **************** FEATURE PIPELINE ******************** //
    float edgeDifference = abs(length(referenceEdgeGradient) - length(testEdgeGradient));
    float pointDifference = abs(length(referencePointGradient) - length(testPointGradient));
    float featureDiff = pow(max(pointDifference, edgeDifference) * M_SQRT1_2, gqf);

    return redistributeErrors(colorDiff, featureDiff);
}

float HDRFLIP(uint2 pixel)
{
    float exposure;
    float ldrflip;
    float hdrflip = 0.0f;
    uint exposureMapIndex = 0u;

    // HDR-FLIP is maximum LDR-FLIP over a range of exposures.
    for (uint i = 0; i < gNumExposures; i++)
    {
        exposure = gStartExposure + i * gExposureDelta;
        ldrflip = LDRFLIP(pixel, exposure);
        if (ldrflip > hdrflip)
        {
            hdrflip = ldrflip;
            exposureMapIndex = i;
        }
    }

    // Store exposure map.
    float exposureMapFloatIndex = float(exposureMapIndex) / (gNumExposures - 1.0f);
    // See reasoning for sRGB2LinearRGB in main().
    gExposureMapDisplay[pixel] = float4(sRGBToLinear(ViridisMap[int(exposureMapFloatIndex * 255.0f + 0.5f)]), 1.0f);

    return hdrflip;
}

float FLIP(uint2 pixel)
{
    if (gIsHDR)
        return HDRFLIP(pixel);
    else
        return LDRFLIP(pixel);
}

[numthreads(32, 32, 1)]
void main(uint3 globalId: SV_DispatchThreadID, uint3 groupThreadId: SV_GroupThreadId)
{
    uint2 pixel = globalId.xy;

    float value = FLIP(pixel);

    if (isnan(value) || isinf(value) || value < 0.0f || value > 1.0f)
    {
        gFLIPErrorMap[pixel] = float4(1.0f, 0.0f, 0.0f, 1.0f);
        gFLIPErrorMapDisplay[pixel] = float4(1.0f, 0.0f, 0.0f, 1.0f);
    }
    else
    {
        // Store FLIP error after possible magma conversion.
        float3 col = gUseMagma ? MagmaMap[int(value * 255.0f + 0.5f)] : float3(value);
        gFLIPErrorMap[pixel] = float4(col, value);

        /*
        Before storing for display, apply an sRGBToLinear transform to make the final output (when displayed)
        the same as what other FLIP implementations generate. This is necessary because, after this shader,
        which stores linear RGB in the FLIP error map, is run, the buffer will be blit:ed to an sRGB buffer.
        The blit includes a linearToSRGB transform. To cancel the effect of that transform, the sRGBToLinear
        transform is added here. This also applies to the exposure map.
        */
        gFLIPErrorMapDisplay[pixel] = float4(sRGBToLinear(col), 1.0f);
    }
}
