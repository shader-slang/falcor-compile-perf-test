/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
#include "DepthOfFieldDefines.slangh"

cbuffer gViewParams
{
    float4x4 ClipToView;
};

cbuffer gDoFCommonParams
{
    uint2 inputSize;
    float2 invInputSize;
    float cameraAnisotropy;
    float subjectDistance;
    float lensScale;
    float splatComputeMaxRadius;
};

SamplerState gBilinearSampler;

Texture2D gInputRadiance;
Texture2D<float> gInputDepth;
RWTexture2D<float4> gOutputRadiance;
Texture2D gOutOfFocusRadiance[BOKEH_MIP_LEVELS];

float square(float f)
{
    return f * f;
}

float2 square(float2 f)
{
    return f * f;
}

// Fast Gaussian texture sample using a 4x4 texel window
float4 gaussianSample(Texture2D tex, SamplerState samp, float2 samplePos, float2 invTextureSize, float sigma = 0.6f)
{
    float2 tc = floor(samplePos - 0.5f) + 0.5f;
    float2 f = tc - samplePos;
    float exponentScale = -0.5f / square(sigma);
    float normalization = 1.f / (sigma * sqrt(2.f * (float)M_PI));

    float2 w0 = normalization * exp(square(f - 1) * exponentScale);
    float2 w1 = normalization * exp(square(f) * exponentScale);
    float2 w2 = normalization * exp(square(f + 1) * exponentScale);
    float2 w3 = normalization * exp(square(f + 2) * exponentScale);

    float2 w01 = w0 + w1;
    float2 w23 = w2 + w3;
    float2 tc01 = invTextureSize * (tc - w0 / w01);
    float2 tc23 = invTextureSize * (tc + 1.f + w3 / w23);

    float4 result = tex.SampleLevel(samp, float2(tc01.x, tc01.y), 0) * w01.x * w01.y +
                    tex.SampleLevel(samp, float2(tc23.x, tc01.y), 0) * w23.x * w01.y +
                    tex.SampleLevel(samp, float2(tc01.x, tc23.y), 0) * w01.x * w23.y +
                    tex.SampleLevel(samp, float2(tc23.x, tc23.y), 0) * w23.x * w23.y;

    return result;
}

float linearizeDepth(float depth)
{
    float4 clipPos = float4(0.f, 0.f, depth, 1.f);
    float4 viewPos = mul(ClipToView, clipPos);
    return abs(viewPos.z / viewPos.w);
}

float computeCoCRadius(float depth)
{
    return clamp(lensScale * (depth - subjectDistance) / depth, -LARGEST_RADIUS, LARGEST_RADIUS);
}

float4 computeCoCRadius(float4 depth)
{
    return float4(computeCoCRadius(depth.x), computeCoCRadius(depth.y), computeCoCRadius(depth.z), computeCoCRadius(depth.w));
}

float getSplatWeight(float radius, float adjustedRadius, float lensScale)
{
    float f = clamp((lensScale - radius) / (LARGEST_RADIUS + lensScale), 0.1f, 1.f);
    f *= f;
    adjustedRadius = max(abs(adjustedRadius), SMALLEST_RADIUS);
    f *= 1.f / ((float)M_PI * adjustedRadius * adjustedRadius);
    return f;
}

[numthreads(16, 16, 1)]
[shader("compute")]
void combineDoF(uint2 pixelIndex: SV_DispatchThreadID)
{
    if (any(float2(pixelIndex) >= inputSize))
        return;

    // Accumulate radiance from the mip chain, start with the full-res level
    float4 radiance = gOutOfFocusRadiance[0][pixelIndex];

    // Go over all the lower-res levels
    float2 samplePos = float2(pixelIndex) + 0.5f;
    float2 sourceSize = (float2)inputSize;

    [unroll]
    for (uint level = 1; level < BOKEH_MIP_LEVELS; level++)
    {
        samplePos *= 0.5f;
        sourceSize = floor(sourceSize * 0.5f);

#if 1
        // Gaussian sampling
        radiance += gaussianSample(gOutOfFocusRadiance[level], gBilinearSampler, samplePos, rcp(sourceSize));
#else
        // Bilinear sampling
        radiance += gOutOfFocusRadiance[level].SampleLevel(gBilinearSampler, samplePos * rcp(sourceSize), 0);
#endif
    }

    // Add the sharp in-focus pixels
    float2 tc = ((float2)pixelIndex + .5f) * invInputSize;
    float depth = gInputDepth.SampleLevel(gBilinearSampler, tc, 0);
    float viewDepth = linearizeDepth(depth);

    float radius = computeCoCRadius(viewDepth);
    if (abs(radius) < 2 * SMALLEST_RADIUS)
    {
        radiance += gInputRadiance[pixelIndex] * getSplatWeight(radius, radius, lensScale);
    }

    // Normalize
    radiance.xyz = radiance.w > 0.f ? radiance.xyz / radiance.w : radiance.xyz;

    // Write out the result
    gOutputRadiance[pixelIndex] = float4(radiance.xyz, 1.f);
}
