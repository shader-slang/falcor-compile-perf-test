/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
import Utils.Math.MathHelpers;
import Helpers;
import LatentTexture;
import SamplingDecoder;

#include "Utils/Math/MathConstants.slangh"

ParameterBlock<SamplingDecoder> gSamplingDecoder;

[BackwardDifferentiable]
float tanhApprox(float x)
{
    return x / sqrt(1.f + x * x);
}

[BackwardDifferentiable]
float sinhApprox(float x)
{
    return x * sqrt(1.f + x * x);
}

[BackwardDifferentiable]
float3 halfvector(float3 wi, float3 wo)
{
    float3 wh = wi + wo;
    wh /= maxFloat(1e-4f, length(wh));
    return wh * sign(wh.z);
}

[BackwardDifferentiable]
float pdfGGX(no_diff float3 wi, no_diff float3 wo, float2 alpha, float rho, float2 mu)
{
    float sqrtOneMinusRhoSqr = sqrt(maxFloat(1e-4f, 1.f - rho * rho));

    // Construct half-vector.
    float3 wh = halfvector(wi, wo);

    // Move to slope-space and revert any slope offset.
    float2 slope = -wh.xy / maxFloat(1e-4f, wh.z);
    slope -= mu;

    // Unstretch slope into alpha=1.0 configuration.
    float sxStd = slope.x / alpha.x;
    float normalization = 1.f / maxFloat(1e-6f, alpha.x * alpha.y * sqrtOneMinusRhoSqr);
    float syStd = (alpha.x * slope.y - rho * alpha.y * slope.x) * normalization;

    // Evaluate basic slope distribution.
    float r2 = sxStd * sxStd + syStd * syStd;
    float p22Std = 1.f / maxFloat(1e-4f, M_PI * (1.f + r2) * (1.f + r2));
    float p22 = p22Std * normalization;
    float pdfH = p22 / maxFloat(1e-4f, wh.z * wh.z * wh.z);

    // Account for specular reflection Jacobian.
    float Jh = 4 * abs(dot(wi, wh));
    return pdfH / maxFloat(1e-4f, Jh);
}

float3 sampleGGX(float3 wi, float2 sample, float2 alpha, float rho, float2 mu)
{
    float sqrtOneMinusRhoSqr = sqrt(maxFloat(1e-4f, 1.f - rho * rho));

    // Sample basic slope.
    float s = sqrt(sample.x) / sqrt(1.f - sample.x);
    float phi = 2.f * M_PI * sample.y;
    float2 slopeStd = s * float2(cos(phi), sin(phi));

    // Stretch slope.
    float2 slope = { alpha.x * slopeStd.x, alpha.y * (rho * slopeStd.x + sqrtOneMinusRhoSqr * slopeStd.y) };

    // Apply slope offset and convert to unit vector.
    slope += mu;
    float3 wh = normalize(float3(-slope.x, -slope.y, 1.f));

    // Reflect along the sampled microfacet.
    return 2.f * dot(wi, wh) * wh - wi;
}

[BackwardDifferentiable]
float pdfDiffuse(no_diff float3 wo, float2 mu)
{
    float3 n = normalize(float3(-mu.x, -mu.y, 1.f));
    float woz = maxFloat(0.f, dot(wo, n));
    return woz / M_PI;
}

float3 sampleDiffuse(float2 sample, float2 mu)
{
    // Sample cosine distribution
    float unused;
    float3 wo = sample_cosine_hemisphere_concentric(sample, unused);

    // Apply normal mapping based on learned slope.
    float3 n = normalize(float3(-mu.x, -mu.y, 1.f));
    float sin = sqrt(maxFloat(1e-4f, 1.f - n.z * n.z));
    float normalization = 1.f / maxFloat(1e-4f, sqrt(maxFloat(1e-4f, n.x * n.x + n.y * n.y)));
    return float3(
        normalization * (n.x * n.z * wo.x - n.y * wo.y + n.x * sin * wo.z),
        normalization * (n.y * n.z * wo.x + n.x * wo.y + n.y * sin * wo.z),
        -sin * wo.x + n.z * wo.z
    );
}

// Abstract importance sampler interface.
interface ILearnedSampler
{
    associatedtype Params;

    [BackwardDifferentiable]
    static Params decodeParams(no_diff float3 wi, no_diff float latentCode[LatentTexture.kNumLatentDims], uint batchIdx, uint warpIdx);

    [BackwardDifferentiable]
    static float pdf(no_diff float3 wi, no_diff float3 wo, Params params);

    static float3 sample(float3 wi, float3 sample, Params params);
}

// 1 Diffuse + 1 GGX lobe importance sampler, driven by MLP that predicts the necessary parameters.
struct DiffuseGGXSampler : ILearnedSampler
{
    struct Params : IDifferentiable
    {
        // Specular parameters.
        float sWeight; // lobe weight
        float2 sAlpha; // anisotropic roughness
        float sRho;    // roughness correlation
        float2 sMu;    // 2d slope offset

        // Diffuse parameters.
        float dWeight; // lobe weight
        float2 dMu;    // 2d slope offset
    };

    [BackwardDifferentiable]
    static Params decodeParams(no_diff float3 wi, no_diff float latentCode[LatentTexture.kNumLatentDims], uint batchIdx, uint warpIdx)
    {
        float output[9];
#ifdef RUN_SAMPLING_DECODER
        // Model input.
        float input[SamplingDecoder.kNumInputs];
        input[0] = wi.x;
        input[1] = wi.y;
        input[2] = wi.z;

        [ForceUnroll]
        for (int i = 0; i < LatentTexture.kNumLatentDims; i++)
        {
            input[3 + i] = latentCode[i];
        }

        // Run network.
        gSamplingDecoder.eval(input, output, batchIdx, warpIdx);
#endif

        // Post-process outputs.
        Params params;
        params.sAlpha = 1e-4f + 0.5f * (1.f + float2(tanhApprox(output[0]), tanhApprox(output[1])));
        params.sRho = tanhApprox(output[2]);
        params.sMu = float2(sinhApprox(output[3]), sinhApprox(output[4]));
        params.dMu = float2(sinhApprox(output[5]), sinhApprox(output[6]));
        params.sWeight = exp(output[7]);
        params.dWeight = exp(output[8]);
        float norm = 1.f / (params.sWeight + params.dWeight);
        params.sWeight *= norm;
        params.dWeight *= norm;

        return params;
    }

    [BackwardDifferentiable]
    static float pdf(no_diff float3 wi, no_diff float3 wo, Params params)
    {
        float pdf = 0.f;
        pdf += params.sWeight * pdfGGX(wi, wo, params.sAlpha, params.sRho, params.sMu);
        pdf += params.dWeight * pdfDiffuse(wo, params.dMu);
        return maxFloat(1e-4f, pdf);
    }

    static float3 sample(float3 wi, float3 sample, Params params)
    {
        if (sample.x < params.sWeight)
            return sampleGGX(wi, sample.yz, params.sAlpha, params.sRho, params.sMu);
        else
            return sampleDiffuse(sample.yz, params.dMu);
    }
};

// Dummy sampler in case we are optimizing without an importance sampler.
struct DummySampler : ILearnedSampler
{
    struct Params : IDifferentiable
    {}

    [BackwardDifferentiable]
    static Params decodeParams(no_diff float3 wi, no_diff float latentCode[LatentTexture.kNumLatentDims], uint batchIdx, uint warpIdx)
    {
        Params params;
        return params;
    }

    [BackwardDifferentiable]
    static float pdf(no_diff float3 wi, no_diff float3 wo, Params params) { return 0.f; }

    static float3 sample(float3 wi, float3 sample, Params params) { return float3(0.f); }
}

#if defined(USE_DIFF_GGX_IMPORTANCE_SAMPLER)
typedef DiffuseGGXSampler LearnedSampler;
#else
typedef DummySampler LearnedSampler;
#endif
