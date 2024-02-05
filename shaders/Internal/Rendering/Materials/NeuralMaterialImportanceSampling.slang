/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"
#include "Internal/Scene/Materials/NeuralMaterialTypes.slangh"

import Utils.Math.MathHelpers;
import Internal.Utils.Neural.Network;
import Scene.Scene;

#if defined(NEURAL_MATERIAL_USE_PROXY_IMPORTANCE_SAMPLING) || defined(NEURAL_MATERIAL_USE_FLOW_IMPORTANCE_SAMPLING)
import NeuralBSDFImportanceSamplingNetwork; // This is a generated source defining the `BSDFImportanceSamplingNetwork` type used below.
#endif

#ifndef NEURAL_MATERIAL_SAMPLING_PRECISION
#error Missing macro definition of NEURAL_MATERIAL_SAMPLING_PRECISION
#endif
#if NEURAL_MATERIAL_SAMPLING_PRECISION == NEURAL_MATERIAL_PRECISION_FP16
typedef float16_t PrecisionTypeIS;
#elif NEURAL_MATERIAL_SAMPLING_PRECISION == NEURAL_MATERIAL_PRECISION_FP32
typedef float PrecisionTypeIS;
#endif

#define NEURAL_MATERIAL_FLOW_MAPPING_CONCENTRIC 0
#define NEURAL_MATERIAL_FLOW_MAPPING_CYLINDRICAL 1

// Importance sampling based on an analytic proxy density.
#ifdef NEURAL_MATERIAL_USE_PROXY_IMPORTANCE_SAMPLING

float3 sampleNeuralBSDFProxy(float3 alpha, float2 slopeSpec, float2 slopeDiff, float weightSpec, float3 wi, float2 u, out float pdf)
{
    float3 wo;
    if (u.x < weightSpec)
    {
        // Specular lobe.
        u.x = u.x / weightSpec;
        wo = sampleSpecular(wi, alpha, slopeSpec, u);
    }
    else
    {
        // Diffuse lobe.
        u.x = (u.x - weightSpec) / (1.f - weightSpec);
        wo = sampleDiffuse(wi, slopeDiff, u);
    }

    pdf = 0.f;
    pdf += weightSpec * pdfSpecular(wi, wo, alpha, slopeSpec);
    pdf += (1.f - weightSpec) * pdfDiffuse(wi, wo, slopeDiff);
    return wo;
}

float evalPdfNeuralBSDFProxy(float3 alpha, float2 slopeSpec, float2 slopeDiff, float weightSpec, float3 wi, float3 wo)
{
    float pdf = 0.f;
    pdf += weightSpec * pdfSpecular(wi, wo, alpha, slopeSpec);
    pdf += (1.f - weightSpec) * pdfDiffuse(wi, wo, slopeDiff);
    return pdf;
}

float tanhApprox(float x)
{
    return x / sqrt(1 + x * x);
}

float sinhApprox(float x)
{
    return x * sqrt(1 + x * x);
}

void evalProxyImportanceSamplingParameters(
    uint networkByteOffset,
    float4 latent0,
    float4 latent1,
    float3 wi,
    out float3 alpha,
    out float2 slopeSpec,
    out float2 slopeDiff,
    out float2 weights
)
{
#if NEURAL_MATERIAL_SAMPLING_PRECISION == NEURAL_MATERIAL_PRECISION_FP16
    StructuredBuffer<NetworkParamStorageFP16> params = gScene.materials.networkParamsFP16;
#elif NEURAL_MATERIAL_SAMPLING_PRECISION == NEURAL_MATERIAL_PRECISION_FP32
    StructuredBuffer<NetworkParamStorageFP32> params = gScene.materials.networkParamsFP32;
#endif

    // Setup network inputs.
#ifdef NEURAL_MATERIAL_PROXY_IS_WI_DEPENDENCE
#if NEURAL_MATERIAL_SAMPLING_INPUT_WIDTH == 11
    PrecisionTypeIS input[11] = {
        wi.x,
        wi.y,
        wi.z,
        latent0.x,
        latent0.y,
        latent0.z,
        latent0.w,
        latent1.x,
        latent1.y,
        latent1.z,
        latent1.w,
    };
#elif NEURAL_MATERIAL_SAMPLING_INPUT_WIDTH == 12
    // Pad inputs to 12 channels. This is a workaround for loading older models that were padded to an even input width.
    PrecisionTypeIS input[12] = {
        wi.x,
        wi.y,
        wi.z,
        latent0.x,
        latent0.y,
        latent0.z,
        latent0.w,
        latent1.x,
        latent1.y,
        latent1.z,
        latent1.w,
        0.f,
    };
#else
#error Unsupported NEURAL_MATERIAL_SAMPLING_INPUT_WIDTH
#endif // NEURAL_MATERIAL_SAMPLING_INPUT_WIDTH
#else
    PrecisionTypeIS input[8] = {
        latent0.x,
        latent0.y,
        latent0.z,
        latent0.w,
        latent1.x,
        latent1.y,
        latent1.z,
        latent1.w,
    };
#endif

    // Setup network outputs.
#if defined(NEURAL_MATERIAL_PROXY_IS_ANISOTROPY) && defined(NEURAL_MATERIAL_PROXY_IS_NORMAL_MAPPING)
    PrecisionTypeIS output[9] = {};
#elif defined(NEURAL_MATERIAL_PROXY_IS_ANISOTROPY) && !defined(NEURAL_MATERIAL_PROXY_IS_NORMAL_MAPPING)
    PrecisionTypeIS output[5] = {};
#elif!defined(NEURAL_MATERIAL_PROXY_IS_ANISOTROPY) && defined(NEURAL_MATERIAL_PROXY_IS_NORMAL_MAPPING)
    PrecisionTypeIS output[7] = {};
#else
    PrecisionTypeIS output[3] = {};
#endif

    BSDFImportanceSamplingNetwork mlp;
    mlp.eval(
        params,
        gScene.materials.networkWeightsTin,
        gScene.materials.networkBiasTin,
        gScene.materials.networkParamsNVCoopVec,
        input,
        output,
        networkByteOffset
    );

    uint k = 0;

    float alphaX = 1e-4f + 0.5f * (1.f + tanhApprox(output[k]));
    k += 1;
    float alphaY = alphaX;
    float rho = 0.f;
#ifdef NEURAL_MATERIAL_PROXY_IS_ANISOTROPY
    alphaY = 1e-4f + 0.5f * (1.f + tanhApprox(output[k]));
    rho = tanhApprox(output[k + 1]);
    k += 2;
#endif

    float slopeSpecX = 0.f;
    float slopeSpecY = 0.f;
    float slopeDiffX = 0.f;
    float slopeDiffY = 0.f;
#ifdef NEURAL_MATERIAL_PROXY_IS_NORMAL_MAPPING
    slopeSpecX = sinhApprox(output[k]);
    slopeSpecY = sinhApprox(output[k + 1]);
    slopeDiffX = sinhApprox(output[k + 2]);
    slopeDiffY = sinhApprox(output[k + 3]);
    k += 4;
#endif

    float wSpec = exp(output[k]);
    float wDiff = exp(output[k + 1]);
    float norm = 1.f / (wSpec + wDiff);
    wSpec *= norm;
    wDiff *= norm;

    alpha = float3(alphaX, alphaY, rho);
    slopeSpec = float2(slopeSpecX, slopeSpecY);
    slopeDiff = float2(slopeDiffX, slopeDiffY);
    weights = float2(wSpec, wDiff);
}

void tangents(float3 n, out float3 s, out float3 t)
{
    const float3 u = float3(1.f, 0.f, 0.f);
    s = normalize(u - n * dot(n, u));
    t = cross(n, s);
}

float3 toWorld(float3 w, float3 n)
{
    float3 s, t;
    tangents(n, s, t);
    return w.x * s + w.y * t + w.z * n;
}

float3 toLocal(float3 w, float3 n)
{
    float3 s, t;
    tangents(n, s, t);
    return float3(dot(w, s), dot(w, t), dot(w, n));
}

float3 sampleDiffuse(float3 wi, float2 slope, float2 u)
{
    // Sample outgoing direction from a cosine distribution.
    float unused;
    float3 wo = sample_cosine_hemisphere_concentric(u, unused);

    // Apply normal mapping.
    float3 n = normalize(float3(-slope.x, -slope.y, 1.f));
    return toWorld(wo, n);
}

float pdfDiffuse(float3 wi, float3 wo, float2 slope)
{
    // Apply normal mapping.
    float3 n = normalize(float3(-slope.x, -slope.y, 1.f));
    wo = toLocal(wo, n);

    // Evaluate cosine density.
    return wo.z * M_1_PI;
}

float3 sampleSpecular(float3 wi, float3 alpha, float2 slope, float2 u)
{
    float rho = alpha.z;
    float sqrtOneMinusRho = sqrt(1.f - rho * rho);

    // Sample base configuration slope.
    float s = sqrt(u.x) / sqrt(1 - u.x);
    float phi = 2 * M_PI * u.y;
    float sxStd = s * cos(phi);
    float syStd = s * sin(phi);

    // Stretch slope based on elliptical anisotropy.
    float sx = alpha.x * sxStd;
    float sy = alpha.y * (rho * sxStd + sqrtOneMinusRho * syStd);

    // Apply the slope offset.
    sx += slope.x;
    sy += slope.y;

    // Convert slope to a half-vector.
    float3 wh = normalize(float3(-sx, -sy, 1.f));
    return 2.f * dot(wi, wh) * wh - wi;
}

float pdfSpecular(float3 wi, float3 wo, float3 alpha, float2 slope)
{
    float rho = alpha.z;
    float sqrtOneMinusRho = sqrt(1.f - rho * rho);

    // Construct half-vector.
    float3 wh = normalize(wi + wo);
    wh *= sign(wh.z);

    float cosTheta = wh.z;
    if (cosTheta <= 1e-4f)
    {
        return 0.f;
    }

    // Convert to slope space.
    float sx = -wh.x / cosTheta;
    float sy = -wh.y / cosTheta;

    // Revert the slope offset.
    sx -= slope.x;
    sy -= slope.y;

    // Unstretch slope based on elliptical anisotropy.
    float sxStd = sx / alpha.x;
    float normalization = 1.f / (alpha.x * alpha.y * sqrtOneMinusRho);
    float syStd = (alpha.x * sy - rho * alpha.y * sx) * normalization;

    // Evaluate basic slope density.
    float r2 = sxStd * sxStd + syStd * syStd;
    float p22Std = 1.f / (M_PI * (1.f + r2) * (1.f + r2));
    float p22 = p22Std * normalization;

    float pdfH = p22 / (cosTheta * cosTheta * cosTheta);

    // Account for specular reflection Jacobian.
    float Jh = 4 * abs(dot(wi, wh));
    return pdfH / Jh;
}

#endif // NEURAL_MATERIAL_USE_PROXY_IMPORTANCE_SAMPLING

// Importance sampling based on normalizing flows.
#ifdef NEURAL_MATERIAL_USE_FLOW_IMPORTANCE_SAMPLING

float3 sampleNeuralBSDFFlow(
    uint networkByteOffset1,
    uint networkByteOffset2,
    float4 latent0,
    float4 latent1,
    float3 wi,
    float2 u,
    out float pdf
)
{
    // Convert `wi` into a 2D quantity.
    float2 wi2 = hemisphereToSquare(wi);

    // Evaluate the flow and tranform resulting 2D sample into a half-vector.
    float pdfSquare;
    float2 wh2 = backwardFlow(u, wi2, latent0, latent1, networkByteOffset1, networkByteOffset2, pdfSquare);
    float3 wh = squareToHemisphere(wh2);

    // Perform a specular reflection.
    float dp = dot(wi, wh);
    float3 wo = 2.f * dp * wh - wi;

    // Compute the corresponding density over outgoing solid angles.
    pdfSquare = 1.f / pdfSquare;
    float pdfH = pdfSquare * M_1_2PI;
    pdf = pdfH / (4.f * abs(dp));

    return wo;
}

float evalPdfNeuralBSDFFlow(uint networkByteOffset1, uint networkByteOffset2, float4 latent0, float4 latent1, float3 wi, float3 wo)
{
    // Compute half-vector and convert it to a 2D position.
    float3 wh = normalize(wi + wo);
    wh *= sign(wh.z);
    float2 wh2 = hemisphereToSquare(wh);

    // Convert `wi` into a 2D quantity.
    float2 wi2 = hemisphereToSquare(wi);

    // Evaluate the flow.
    float pdfSquare;
    forwardFlow(wh2, wi2, latent0, latent1, networkByteOffset1, networkByteOffset2, pdfSquare);

    // Compute the corresponding density over outgoing solid angles.
    float pdfH = pdfSquare * M_1_2PI;
    return pdfH / (4.f * abs(dot(wi, wh)));
}

float3 squareToHemisphere(float2 p)
{
#if NEURAL_MATERIAL_FLOW_MAPPING == NEURAL_MATERIAL_FLOW_MAPPING_CONCENTRIC
    // Concentric disk mapping.
    p = 2.f * p - 1.f;
    float r, rp, phi;
    if (p.x == 0.f && p.y == 0.f)
    {
        r = 0.f;
        rp = 0.f;
        phi = 0.f;
    }
    else if (abs(p.x) > abs(p.y))
    {
        // Quadrant 0 or 2
        r = p.x;
        rp = p.y;
        phi = 0.25f * M_PI * rp / r;
    }
    else
    {
        // Quadrant 1 or 3
        r = p.y;
        rp = p.x;
        phi = 0.25f * M_PI * rp / r;
        phi = 0.5f * M_PI - phi;
    }

    float2 pDisk = r * float2(cos(phi), sin(phi));
    float z = 1.f - (pDisk.x * pDisk.x + pDisk.y * pDisk.y);
    pDisk = pDisk * sqrt(z + 1.f);
    return float3(pDisk.x, pDisk.y, z);
#elif NEURAL_MATERIAL_FLOW_MAPPING == NEURAL_MATERIAL_FLOW_MAPPING_CYLINDRICAL
    float mu = p.x;
    float phi = 2.f * M_PI * p.y - M_PI;
    float sinTheta = sqrt(1.f - mu * mu);
    return float3(sinTheta * cos(phi), sinTheta * sin(phi), mu);
#else
#error Missing macro definition of NEURAL_MATERIAL_FLOW_MAPPING
#endif
}

float2 hemisphereToSquare(float3 w)
{
#if NEURAL_MATERIAL_FLOW_MAPPING == NEURAL_MATERIAL_FLOW_MAPPING_CONCENTRIC
    float2 p = float2(w.x, w.y);
    p /= sqrt(w.z + 1.0);

    // Inverse concentric disk mapping.
    bool quad0Or2 = abs(p.x) > abs(p.y);
    float rSign = quad0Or2 ? p.x : p.y;
    float sg = sign(rSign);
    float r = length(p) * sg;

    float phi = atan2(sg * p.y, sg * p.x);

    float t = 4.0f * M_1_PI * phi;
    t = (quad0Or2 ? t : 2.f - t) * r;
    float a = quad0Or2 ? r : t;
    float b = quad0Or2 ? t : r;

    p = float2(a, b);
    return 0.5f * (p + 1.f);
#elif NEURAL_MATERIAL_FLOW_MAPPING == NEURAL_MATERIAL_FLOW_MAPPING_CYLINDRICAL
    float x = w.z;
    float phi = atan2(w.y, w.x);
    float y = M_1_2PI * (phi + M_PI);
    return float2(x, y);
#else
#error Missing macro definition of NEURAL_MATERIAL_FLOW_MAPPING
#endif
}

float2 forwardFlow(float2 x, float2 wi2, float4 latent0, float4 latent1, uint networkByteOffset1, uint networkByteOffset2, out float pdf)
{
    float x0 = x.x;
    float x1 = x.y;

    pdf = 1.f;
    float p;

    // Coupling layer 1.
    float y0 = warpPiecewiseQuadratic(x1, x0, wi2, latent0, latent1, networkByteOffset1, p);
    pdf *= p;

    // Coupling layer 2.
    float y1 = warpPiecewiseQuadratic(x0, y0, wi2, latent0, latent1, networkByteOffset2, p);
    pdf *= p;

    return float2(y0, y1);
}

float2 backwardFlow(float2 y, float2 wi2, float4 latent0, float4 latent1, uint networkByteOffset1, uint networkByteOffset2, out float pdf)
{
    float y0 = y.x;
    float y1 = y.y;

    pdf = 1.f;
    float p;

    // Coupling layer 2.
    float x0 = unwarpPiecewiseQuadratic(y1, y0, wi2, latent0, latent1, networkByteOffset2, p);
    pdf *= p;

    // Coupling layer 1.
    float x1 = unwarpPiecewiseQuadratic(y0, x0, wi2, latent0, latent1, networkByteOffset1, p);
    pdf *= p;

    return float2(x0, x1);
}

struct PiecewiseQuadraticBins
{
    float dx[NEURAL_MATERIAL_FLOW_N_BINS - 1];
    float xs[NEURAL_MATERIAL_FLOW_N_BINS - 1];
    float ys[NEURAL_MATERIAL_FLOW_N_BINS - 1];
    float vertices[NEURAL_MATERIAL_FLOW_N_BINS];
};

float warpPiecewiseQuadratic(float x, float x_unchanged, float2 wi2, float4 latent0, float4 latent1, uint networkByteOffset, out float pdf)
{
    const uint nBins = NEURAL_MATERIAL_FLOW_N_BINS;

    PiecewiseQuadraticBins bins;
    evalWarpParameters(networkByteOffset, x_unchanged, wi2, latent0, latent1, bins);

    uint b = 0;
    for (uint i = 0; i < nBins - 2; ++i)
    {
        if (x >= bins.xs[i] && x < bins.xs[i + 1])
        {
            b = i + 1;
            break;
        }
    }

    float vertexPrev = bins.vertices[b];
    float vertex = bins.vertices[b + 1];

    float xsPrev = 0.f;
    float ysPrev = 0.f;
    if (b > 0)
    {
        xsPrev = bins.xs[b - 1];
        ysPrev = bins.ys[b - 1];
    }

    float slope = (vertex - vertexPrev) / bins.dx[b];
    float xdiff = x - xsPrev;
    float y = ysPrev + 0.5f * xdiff * xdiff * slope + vertexPrev * xdiff;

    pdf = vertexPrev + xdiff * slope;
    return clamp(y, 0.f, 1.f);
}

float unwarpPiecewiseQuadratic(
    float y,
    float y_unchanged,
    float2 wi2,
    float4 latent0,
    float4 latent1,
    uint networkByteOffset,
    out float pdf
)
{
    const uint nBins = NEURAL_MATERIAL_FLOW_N_BINS;

    PiecewiseQuadraticBins bins;
    evalWarpParameters(networkByteOffset, y_unchanged, wi2, latent0, latent1, bins);

    uint b = 0;
    for (uint i = 0; i < nBins - 2; ++i)
    {
        if (y >= bins.ys[i] && y < bins.ys[i + 1])
        {
            b = i + 1;
            break;
        }
    }

    float vertexPrev = bins.vertices[b];
    float vertex = bins.vertices[b + 1];

    float xsPrev = 0.f;
    float ysPrev = 0.f;
    if (b > 0)
    {
        xsPrev = bins.xs[b - 1];
        ysPrev = bins.ys[b - 1];
    }

    float slope = (vertex - vertexPrev) / bins.dx[b];
    float ydiff = y - ysPrev;
    float xdiff = 2.f * ydiff / (vertexPrev + sqrt(vertexPrev * vertexPrev + 2 * slope * ydiff));
    float x = xsPrev + xdiff;

    pdf = 1.f / (vertexPrev + xdiff * slope);
    return clamp(x, 0.f, 1.f);
}

void evalWarpParameters(
    uint networkByteOffset,
    float unchanged,
    float2 wi2,
    float4 latent0,
    float4 latent1,
    out PiecewiseQuadraticBins bins
)
{
    const uint nBins = NEURAL_MATERIAL_FLOW_N_BINS;

    PrecisionTypeIS input[12] = {
        unchanged,
        wi2.x,
        wi2.y,
        latent0.x,
        latent0.y,
        latent0.z,
        latent0.w,
        latent1.x,
        latent1.y,
        latent1.z,
        latent1.w,
        0.f,
    };

#if NEURAL_MATERIAL_SAMPLING_PRECISION == NEURAL_MATERIAL_PRECISION_FP16
    StructuredBuffer<NetworkParamStorageFP16> params = gScene.materials.networkParamsFP16;
#elif NEURAL_MATERIAL_SAMPLING_PRECISION == NEURAL_MATERIAL_PRECISION_FP32
    StructuredBuffer<NetworkParamStorageFP32> params = gScene.materials.networkParamsFP32;
#endif

    PrecisionTypeIS output[2 * NEURAL_MATERIAL_FLOW_N_BINS - 1] = {};

    BSDFImportanceSamplingNetwork mlp;
    mlp.eval(
        params,
        gScene.materials.networkWeightsTin,
        gScene.materials.networkBiasTin,
        gScene.materials.networkParamsNVCoopVec,
        input,
        output,
        networkByteOffset
    );

    const float regPdf = 0.01f;
    const float regSpacing = 0.01f;

    // Softmax on dx
    float normalization = 0.f;
    for (int i = 0; i < nBins - 1; ++i)
    {
        float tmp = exp(output[i]);
        normalization += tmp;
        bins.dx[i] = tmp;
    }
    normalization = 1.f / normalization;
    for (int i = 0; i < nBins - 1; ++i)
    {
        bins.dx[i] *= normalization;
        bins.dx[i] = bins.dx[i] * (1.f - regSpacing) + regSpacing / (float(nBins) - 1.f);
    }

    // Integrate spacings.
    bins.xs[0] = bins.dx[0];
    for (int i = 0; i < nBins - 2; ++i)
    {
        bins.xs[i + 1] = bins.xs[i] + bins.dx[i + 1];
    }
    bins.xs[nBins - 2] = 1.000001f;

    // Vertices and their integrals.
    normalization = 0.f;
    for (int i = 0; i < nBins; ++i)
    {
        float tmp = exp(output[nBins - 1 + i]);
        normalization += tmp;
        bins.vertices[i] = tmp;
    }
    normalization = 1.f / normalization;
    for (int i = 0; i < nBins; ++i)
    {
        bins.vertices[i] *= normalization;
    }

    float dy[NEURAL_MATERIAL_FLOW_N_BINS - 1];
    normalization = 0.f;
    for (int i = 0; i < nBins - 1; ++i)
    {
        float scale = 0.5f * (bins.vertices[i] + bins.vertices[i + 1]);
        float tmp = bins.dx[i] * scale;
        normalization += tmp;
        dy[i] = tmp;
    }
    normalization = 1.f / normalization;

    for (int i = 0; i < nBins; ++i)
    {
        if (i < nBins - 1)
        {
            dy[i] *= normalization;
            dy[i] = dy[i] * (1.f - regPdf) + regPdf * bins.dx[i];
        }
        bins.vertices[i] *= normalization;
        bins.vertices[i] = bins.vertices[i] * (1.f - regPdf) + regPdf;
    }

    bins.ys[0] = dy[0];
    for (int i = 0; i < nBins - 2; ++i)
    {
        bins.ys[i + 1] = bins.ys[i] + dy[i + 1];
    }
    bins.ys[nBins - 2] = 1.000001f;
}

#endif // NEURAL_MATERIAL_USE_FLOW_IMPORTANCE_SAMPLING
