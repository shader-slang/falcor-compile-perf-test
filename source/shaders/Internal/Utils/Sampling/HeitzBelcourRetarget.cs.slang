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
Texture2D<float4> gBlueNoise2Dx1DRetargetingTextures[kNumberOfBNTextures];
Texture2D<float4> gRender;
RWTexture2D<uint> gSeedsSrc;
RWTexture2D<uint> gSeedsDst;

static const NoiseTarget kNoiseTarget = NoiseTarget(NOISE_TARGET);
static const bool kRetarget = bool(RETARGET);

float2 readRetargetingTexture(uint2 px)
{
    switch (kNoiseTarget)
    {
    case NoiseTarget::BlueNoise2D:
        return gBlueNoise2DRetargetingTexture[px].rg;
    case NoiseTarget::BlueNoise2Dx1D:
        return gBlueNoise2Dx1DRetargetingTextures[gFrameIndex % kNumberOfBNTextures][px].rg;
    case NoiseTarget::IGN: // IGN doesn't support retargetting.
    default:
        return float2(0.f);
    }
}

uint2 getRetargetingTextureDims()
{
    uint2 ret = uint2(0, 0);
    switch (kNoiseTarget)
    {
    case NoiseTarget::BlueNoise2D:
        gBlueNoise2DRetargetingTexture.GetDimensions(ret.x, ret.y);
        break;
    case NoiseTarget::BlueNoise2Dx1D:
        gBlueNoise2Dx1DRetargetingTextures[gFrameIndex % kNumberOfBNTextures].GetDimensions(ret.x, ret.y);
        break;
    case NoiseTarget::IGN:
        break; // IGN doesn't support retargetting.
    default:
        break;
    }
    return ret;
}

float2 R2(uint index)
{
    // Generalized golden ratio to 2d.
    // Solution to x^3 = x + 1
    // AKA plastic constant.
    // from http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    float g = 1.32471795724474602596f;
    return frac(float2(float(index) / g, float(index) / (g * g)));
}

uint2 retarget(uint2 pos, uint2 retargetDims, uint frameIndex)
{
    uint2 frameOffset = (kNoiseTarget != NoiseTarget::BlueNoise2Dx1D) ? uint2(R2(frameIndex) * float2(retargetDims - 1)) : uint2(0, 0);
    float2 retargetF = readRetargetingTexture((pos + frameOffset) % retargetDims).rg;
    int2 retargetI = int2(retargetF * 255.0f) - 6;
    return uint2(int2(pos) + retargetI);
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    // get dimensions of the seed texture and retargetting texture
    uint2 seedDim;
    gSeedsDst.GetDimensions(seedDim.x, seedDim.y);

    uint2 retargetDims = getRetargetingTextureDims();

    // get the bounds for our block
    uint2 blockIndex = dispatchThreadId.xy;
    uint2 blockStart = blockIndex * BLOCK_WIDTH;
    uint2 blockEnd = (blockIndex + 1) * BLOCK_WIDTH;
    if (any(blockEnd.xy > seedDim))
        return;

    // retarget
    for (uint iy = blockStart.y; iy < blockEnd.y; ++iy)
    {
        for (uint ix = blockStart.x; ix < blockEnd.x; ++ix)
        {
            uint2 writePos = uint2(ix, iy);

            if (kRetarget)
                writePos = retarget(uint2(ix, iy), retargetDims, gFrameIndex);

            gSeedsDst[writePos % seedDim] = gSeedsSrc[uint2(ix, iy) % seedDim];
        }
    }

#if 0
    // NOTE: this works! the difference is very low as expected! Leaving it here as it helps debugging
    // "thisFrameValue" -> the blue noise value at this pixel this frame
    int2 frameScrolling = int2(R2(gFrameIndex) * float2(retargetDims-1));
    int2 scrolledPosition = int2(ix, iy) + frameScrolling;
    uint thisFrameValue = uint(gBlueNoise2DTexture[scrolledPosition % retargetDims] * 256.0f);

    // "nextFrameValue" -> the value next frame, at where this pixel is retargetted to.
    int2 retargettingOffset = (int2(readRetargetingTexture(scrolledPosition % retargetDims).rg * 256.0f) - 6);
    int2 scrollToNextFrame = int2(R2(1) * float2(retargetDims-1));
    uint nextFrameValue = uint(gBlueNoise2DTexture[(scrolledPosition+retargettingOffset+scrollToNextFrame) %
    retargetDims] * 256.0f);

    int diff = int(thisFrameValue) - int(nextFrameValue);
    gSeedsSrc[uint2(ix,iy)] = abs(diff);
#endif
}
