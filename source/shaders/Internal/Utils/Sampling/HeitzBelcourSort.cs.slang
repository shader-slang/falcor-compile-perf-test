/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/

/**
 * Compute shader that moves seed values around to make a render more like a target pattern
 */

import HeitzBelcourSampleGeneratorTypes;
import Utils.Color.ColorHelpers;
import Utils.Math.HashUtils;
import Utils.Math.BitTricks;

// BLOCK_WIDTH is defined by the compiling code, to something like 4.
#define BLOCK_SIZE (BLOCK_WIDTH * BLOCK_WIDTH)

cbuffer PerFrameCB
{
    uint2 gFrameDim;
    uint gFrameIndex;
}

// Textures and UAVs
static const uint kNumberOfBNTextures = 64;
Texture2D<float> gBlueNoise2DTexture;
Texture2D<float4> gBlueNoise2DRetargetingTexture;
Texture2D<float> gBlueNoise2Dx1DTextures[kNumberOfBNTextures];
Texture2D<float> gBlueNoise2Dx1DRetargetingTextures[kNumberOfBNTextures];
Texture2D<float4> gRender;
RWTexture2D<uint> gSeedsSrc;
RWTexture2D<uint> gSeedsDst;

static const NoiseTarget kNoiseTarget = NoiseTarget(NOISE_TARGET);
static const bool kRetarget = bool(RETARGET);

float2 R2(uint index)
{
    // Generalized golden ratio to 2d.
    // Solution to x^3 = x + 1
    // AKA plastic constant.
    // from http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    float g = 1.32471795724474602596f;
    return frac(float2(float(index) / g, float(index) / (g * g)));
}

// Interleaved Gradient Noise - a spatial low discrepancy sequence with good properties under TAA that does 3x3
// neighborhood sampling for history rejection. Talked about in
// http://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare Presented at siggraph in 2014
// in http://advances.realtimerendering.com/s2014/ part 1.
float IGN(uint2 pos)
{
    return frac(52.9829189f * frac(0.06711056f * float(pos.x) + 0.00583715f * float(pos.y)));
}

float readTarget(uint2 pos)
{
    switch (kNoiseTarget)
    {
    case NoiseTarget::BlueNoise2D:
    {
        uint w, h;
        gBlueNoise2DTexture.GetDimensions(w, h);
        float2 offset = R2(gFrameIndex) * float2(w - 1, h - 1);
        return gBlueNoise2DTexture[(int2(pos) + int2(offset)) % uint2(w, h)];
    }
    case NoiseTarget::BlueNoise2Dx1D:
    {
        uint w, h;
        gBlueNoise2Dx1DTextures[gFrameIndex % kNumberOfBNTextures].GetDimensions(w, h);
        return gBlueNoise2Dx1DTextures[gFrameIndex % kNumberOfBNTextures][pos % uint2(w, h)];
    }
    case NoiseTarget::IGN:
    {
        uint2 offset = uint2(R2(gFrameIndex) * 500.0f);
        return IGN(pos + offset);
    }
    default:
        return 0.f;
    }
}

void sort(inout float luma[BLOCK_SIZE], inout uint pos[BLOCK_SIZE])
{
    // Sorting adapted from https://en.wikipedia.org/wiki/Bitonic_sorter
    static const int n = BLOCK_SIZE;

    for (int k = 2; k <= n; k *= 2) // k is doubled every iteration
    {
        for (int j = k / 2; j > 0; j /= 2) // j is halved at every iteration, with truncation of fractional parts
        {
            for (int i = 0; i < n; i++)
            {
                int l = i ^ j;
                if (l > i)
                {
                    float lumai = luma[i];
                    float lumal = luma[l];

                    if (((i & k) == 0 && lumai > lumal) || ((i & k) != 0 && lumai < lumal))
                    {
                        luma[i] = lumal;
                        luma[l] = lumai;

                        uint posi = pos[i];
                        uint posl = pos[l];
                        pos[i] = posl;
                        pos[l] = posi;
                    }
                }
            }
        }
    }
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    uint2 seedDim;
    gSeedsDst.GetDimensions(seedDim.x, seedDim.y);

    // get the bounds for our block
    uint2 blockIndex = dispatchThreadId.xy;
    ;
    uint2 blockStart = blockIndex * BLOCK_WIDTH;
    uint2 blockEnd = (blockIndex + 1) * BLOCK_WIDTH;
    if (any(blockEnd.xy > seedDim))
        return;

    // Sort render and target, move the seeds to make the next render be more like the target.
    // This is real inneficient but I'm short on time and I need image results, not perf.

    // Load render data
    float renderLuma[BLOCK_SIZE];
    uint renderPos[BLOCK_SIZE];
    for (uint iy = blockStart.y; iy < blockEnd.y; ++iy)
    {
        for (uint ix = blockStart.x; ix < blockEnd.x; ++ix)
        {
            int flatIndex = (iy - blockStart.y) * BLOCK_WIDTH + (ix - blockStart.x);
            renderLuma[flatIndex] = dot(gRender[uint2(ix, iy)].rgb, float3(0.3f, 0.59f, 0.11f));
            renderPos[flatIndex] = flatIndex;
        }
    }

    // Sort render data
    sort(renderLuma, renderPos);

    // Load target data
    float targetLuma[BLOCK_SIZE];
    uint targetPos[BLOCK_SIZE];
    for (uint iy = blockStart.y; iy < blockEnd.y; ++iy)
    {
        for (uint ix = blockStart.x; ix < blockEnd.x; ++ix)
        {
            int flatIndex = (iy - blockStart.y) * BLOCK_WIDTH + (ix - blockStart.x);
            targetLuma[flatIndex] = readTarget(uint2(ix, iy));
            targetPos[flatIndex] = flatIndex;
        }
    }

    // Sort target data
    sort(targetLuma, targetPos);

    // Write out seeds in the order that would make them more like the target
    for (uint index = 0; index < BLOCK_SIZE; ++index)
    {
        uint2 renderPosOffset;
        renderPosOffset.x = renderPos[index] % BLOCK_WIDTH;
        renderPosOffset.y = renderPos[index] / BLOCK_WIDTH;

        uint2 targetPosOffset;
        targetPosOffset.x = targetPos[index] % BLOCK_WIDTH;
        targetPosOffset.y = targetPos[index] / BLOCK_WIDTH;

        uint seed = gSeedsSrc[blockStart + renderPosOffset];

        gSeedsDst[(blockStart + targetPosOffset) % seedDim] = seed;
    }
}
