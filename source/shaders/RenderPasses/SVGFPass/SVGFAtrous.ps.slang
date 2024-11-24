/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import Utils.Math.MathHelpers;
import Utils.Color.ColorHelpers;
import SVGFCommon;

cbuffer PerImageCB
{
    Texture2D gAlbedo;
    Texture2D gHistoryLength;
    Texture2D gIllumination;
    Texture2D gLinearZAndNormal;

    int gStepSize;
    float gPhiColor;
    float gPhiNormal;
};

// computes a 3x3 gaussian blur of the variance, centered around
// the current pixel
float computeVarianceCenter(int2 ipos)
{
    float sum = 0.f;

    const float kernel[2][2] = {
        { 1.0 / 4.0, 1.0 / 8.0 },
        { 1.0 / 8.0, 1.0 / 16.0 },
    };

    const int radius = 1;
    for (int yy = -radius; yy <= radius; yy++)
    {
        for (int xx = -radius; xx <= radius; xx++)
        {
            const int2 p = ipos + int2(xx, yy);
            const float k = kernel[abs(xx)][abs(yy)];
            sum += gIllumination.Load(int3(p, 0)).a * k;
        }
    }

    return sum;
}

float4 main(FullScreenPassVsOut vsOut) : SV_TARGET0
{
    const int2 ipos = int2(vsOut.posH.xy);
    const int2 screenSize = getTextureDims(gAlbedo, 0);

    const float epsVariance = 1e-10;
    const float kernelWeights[3] = { 1.0, 2.0 / 3.0, 1.0 / 6.0 };

    // constant samplers to prevent the compiler from generating code which
    // fetches the sampler descriptor from memory for each texture access
    const float4 illuminationCenter = gIllumination.Load(int3(ipos, 0));
    const float lIlluminationCenter = luminance(illuminationCenter.rgb);

    // variance, filtered using 3x3 gaussin blur
    const float var = computeVarianceCenter(ipos);

    // number of temporally integrated pixels
    const float historyLength = gHistoryLength[ipos].x;

    const float2 zCenter = gLinearZAndNormal[ipos].xy;
    if (zCenter.x < 0)
    {
        // not a valid depth => must be envmap => do not filter
        return illuminationCenter;
    }
    const float3 nCenter = oct_to_ndir_snorm(gLinearZAndNormal[ipos].zw);

    const float phiLIllumination = gPhiColor * sqrt(max(0.0, epsVariance + var.r));
    const float phiDepth = max(zCenter.y, 1e-8) * gStepSize;

    // explicitly store/accumulate center pixel with weight 1 to prevent issues
    // with the edge-stopping functions
    float sumWIllumination = 1.0;
    float4 sumIllumination = illuminationCenter;

    for (int yy = -2; yy <= 2; yy++)
    {
        for (int xx = -2; xx <= 2; xx++)
        {
            const int2 p = ipos + int2(xx, yy) * gStepSize;
            const bool inside = all(p >= int2(0, 0)) && all(p < screenSize);

            const float kernel = kernelWeights[abs(xx)] * kernelWeights[abs(yy)];

            if (inside && (xx != 0 || yy != 0)) // skip center pixel, it is already accumulated
            {
                const float4 illuminationP = gIllumination.Load(int3(p, 0));
                const float lIlluminationP = luminance(illuminationP.rgb);
                const float zP = gLinearZAndNormal[p].x;
                const float3 nP = oct_to_ndir_snorm(gLinearZAndNormal[p].zw);

                // compute the edge-stopping functions
                const float2 w = computeWeight(
                    zCenter.x,
                    zP,
                    phiDepth * length(float2(xx, yy)),
                    nCenter,
                    nP,
                    gPhiNormal,
                    lIlluminationCenter,
                    lIlluminationP,
                    phiLIllumination
                );

                const float wIllumination = w.x * kernel;

                // alpha channel contains the variance, therefore the weights need to be squared, see paper for the formula
                sumWIllumination += wIllumination;
                sumIllumination += float4(wIllumination.xxx, wIllumination * wIllumination) * illuminationP;
            }
        }
    }

    // renormalization is different for variance, check paper for the formula
    float4 filteredIllumination = float4(sumIllumination / float4(sumWIllumination.xxx, sumWIllumination * sumWIllumination));

    return filteredIllumination;
}
