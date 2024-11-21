/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/

/** List of alpha testing modes.
*/
enum AlphaTestMode
{
    Disabled,
    Basic,
    HashedIsotropic,
    HashedAnisotropic,
};

/** Default to basic thresholding.
    This works for both raster and raytrace passes.
*/
#ifndef _ALPHA_TEST_MODE
#define _ALPHA_TEST_MODE (uint)AlphaTestMode::Basic
#endif

static const AlphaTestMode kAlphaTestMode = AlphaTestMode(_ALPHA_TEST_MODE);

/** Evaluates alpha testing.
    \param[in] alpha Alpha value to test [0..1].
    \param[in] threshold Alpha threshold value [0..1].
    \param[in] posW World-space shading position.
    \return Returns true if pixel should be discarded (i.e. surface is non-opaque).
*/
bool evalAlphaTest(float alpha, float threshold, float3 posW)
{
    float hashedAlphaScale = 1.0f;
#ifdef _HASHED_ALPHA_SCALE
    hashedAlphaScale = _HASHED_ALPHA_SCALE;
#endif

    if (kAlphaTestMode == AlphaTestMode::Basic)
    {
        return evalBasicAlphaTest(alpha, threshold);
    }
    else if (kAlphaTestMode == AlphaTestMode::HashedIsotropic)
    {
        float hashedThreshold = calculateHashedAlpha(posW, hashedAlphaScale, false);
        return evalHashedAlphaTest(alpha, threshold, hashedThreshold);
    }
    else if (kAlphaTestMode == AlphaTestMode::HashedAnisotropic)
    {
        float hashedThreshold = calculateHashedAlpha(posW, hashedAlphaScale, true);
        return evalHashedAlphaTest(alpha, threshold, hashedThreshold);
    }

    return false;
}

/** Evaluates alpha testing using hard thresholding.
*/
bool evalBasicAlphaTest(float alpha, float threshold)
{
    return alpha < threshold;
}

/** Evaluates alpha testing using hashed thresholding.
*/
bool evalHashedAlphaTest(float alpha, float materialThreshold, float hashedThreshold)
{
    float compareTo = hashedThreshold <= 0 ? materialThreshold : saturate(hashedThreshold);
    return alpha < compareTo;
}

float sineHash(float2 p)
{
    return frac(1e4 * sin(17.0 * p.x + p.y * 0.1) * (0.1 + abs(sin(p.y * 13.0 + p.x))));
}

float sineHash3D(float3 p)
{
    return sineHash(float2(sineHash(p.xy), p.z));
}

float calculateHashedAlpha(float3 hashInputCoord, float hashScale, bool useAnisotropy)
{
    // Provide a decent default to our alpha threshold
    float alphaCompare = 0.5f;

    if (useAnisotropy)
    {
        //////  Anisotropic version

        // Find the discretized derivatives of our coordinates
        float3 anisoDeriv = max(abs(ddx(hashInputCoord)), abs(ddy(hashInputCoord)));
        float3 anisoScales = float3(0.707f / (hashScale * anisoDeriv.x),
                                    0.707f / (hashScale * anisoDeriv.y),
                                    0.707f / (hashScale * anisoDeriv.z));
        // Find log-discretized noise scales
        float3 scaleFlr = float3(exp2(floor(log2(anisoScales.x))),
                                 exp2(floor(log2(anisoScales.y))),
                                 exp2(floor(log2(anisoScales.z))));
        float3 scaleCeil = float3(exp2(ceil(log2(anisoScales.x))),
                                  exp2(ceil(log2(anisoScales.y))),
                                  exp2(ceil(log2(anisoScales.z))));
        // Compute alpha thresholds at our two noise scales
        float2 alpha = float2(sineHash3D(floor(scaleFlr * hashInputCoord)),
                              sineHash3D(floor(scaleCeil * hashInputCoord)));
        // Factor to linearly interpolate with
        float3 fractLoc = float3(frac(log2(anisoScales.x)),
                                 frac(log2(anisoScales.y)),
                                 frac(log2(anisoScales.z)));
        float2 toCorners = float2(length(fractLoc),
                                  length(float3(1.0f, 1.0f, 1.0f) - fractLoc));
        float lerpFactor = toCorners.x / (toCorners.x + toCorners.y);
        // Interpolate alpha threshold from noise at two scales
        float x = (1 - lerpFactor) * alpha.x + lerpFactor * alpha.y;
        // Pass into CDF to compute uniformly distrib threshold
        float a = min(lerpFactor, 1 - lerpFactor);
        float3 cases = float3(x * x / (2 * a * (1 - a)), (x - 0.5 * a) / (1 - a), 1.0 - ((1 - x) * (1 - x) / (2 * a * (1 - a))));
        // Find our final, uniformly distributed alpha threshold
        alphaCompare = (x < (1 - a)) ? ((x < a) ? cases.x : cases.y) : cases.z;
        alphaCompare = clamp(alphaCompare, 1.0e-6, 1.0f);
    }
    else
    {
        //////  Isotropic version

        // Find the discretized derivatives of our coordinates
        float maxDeriv = max(length(ddx(hashInputCoord)), length(ddy(hashInputCoord)));
        float pixScale = 1.0 / (hashScale * maxDeriv);
        // Find two nearest log-discretized noise scales
        float2 pixScales = float2(exp2(floor(log2(pixScale))), exp2(ceil(log2(pixScale))));
        // Compute alpha thresholds at our two noise scales
        float2 alpha = float2(sineHash3D(floor(pixScales.x * hashInputCoord)), sineHash3D(floor(pixScales.y * hashInputCoord)));
        // Factor to interpolate lerp with
        float lerpFactor = frac(log2(pixScale));
        // Interpolate alpha threshold from noise at two scales
        float x = (1 - lerpFactor) * alpha.x + lerpFactor * alpha.y;
        float a = min(lerpFactor, 1 - lerpFactor);
        // Pass into CDF to compute uniformly distrib threshold
        float3 cases = float3(x * x / (2 * a * (1 - a)), (x - 0.5 * a) / (1 - a), 1.0 - ((1 - x) * (1 - x) / (2 * a * (1 - a))));
        // Find our final, uniformly distributed alpha threshold
        alphaCompare = (x < (1 - a)) ? ((x < a) ? cases.x : cases.y) : cases.z;
        alphaCompare = clamp(alphaCompare, 1e-6f, 1.0f);
    }

    return alphaCompare;
}
