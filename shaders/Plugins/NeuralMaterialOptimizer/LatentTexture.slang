/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
#include "Utils/NVAPI.slangh" // For atomics
import Utils.Sampling.SampleGenerator;

struct LatentTexture
{
    static const uint32_t kNumLatentDims = NUM_LATENT_DIMS;
    static const uint32_t kWidth = LATENT_TEXTURE_WIDTH;
    static const uint32_t kHeight = LATENT_TEXTURE_HEIGHT;
    static const uint32_t kNumUdimTiles = NUM_UDIM_TILES;
    static const uint32_t kNumMipLevels = NUM_MIP_LEVELS;

    RWTexture2D<float> parameters[kNumUdimTiles][kNumMipLevels][kNumLatentDims];
    RWTexture2D<float> derivatives[kNumUdimTiles][kNumMipLevels][kNumLatentDims];
    RWTexture2D<float2> moments[kNumUdimTiles][kNumMipLevels][kNumLatentDims];
    StructuredBuffer<int> udimIndirection;

    uint uv2index(float2 uv)
    {
        if (kNumUdimTiles == 1)
            return 0; // To correctly handle arbitrary UV ranges (with wrapping) for non-udim materials.

        uint2 tileCoord = max(int2(uv), int2(0));
        uint udimID = tileCoord.x + 10 * tileCoord.y;

        // udimIndirection[udimID] is -1 for some tiles, but we should never
        // receive uvs for those. Clamp to 0 just in case.
        // assert(udimIndirection[udimID] != -1, "Error: UVs are accessing an inactive tile (with no textures).");
        uint texID = max(0, udimIndirection[udimID]);

        return texID;
    }

    [BackwardDifferentiable]
    void getCodeNearest(float2 uv, out float code[kNumLatentDims])
    {
        uint tileIndex = uv2index(uv);
        uv = frac(uv);
        float2 uv_scaled = uv * float2(kWidth, kHeight);
        uv_scaled = clamp(uv_scaled, float2(0.f, 0.f), float2(kWidth - 1, kHeight - 1));
        uint2 texelId = uint2(uint(uv_scaled[0]), uint(uv_scaled[1]));

        [ForceUnroll]
        for (int ch = 0; ch < kNumLatentDims; ch++)
        {
            code[ch] = getTexel(tileIndex, 0, ch, texelId);
        }
    }

    [BackwardDifferentiable]
    void getCodeBilinear(float2 uv, uint mipLevel, out float code[kNumLatentDims])
    {
        uint tileIndex = uv2index(uv);
        uv = frac(uv);

        uint width = kWidth >> mipLevel;
        uint height = kHeight >> mipLevel;
        float2 uvScaled = uv * float2(width, height) - float2(0.5f, 0.5f);

        uint u0 = clamp(uint(uvScaled[0]), 0, width - 1);
        uint u1 = u0 < width - 1 ? u0 + 1 : 0;
        uint v0 = clamp(uint(uvScaled[1]), 0, height - 1);
        uint v1 = v0 < height - 1 ? v0 + 1 : 0;

        float2 uvFrac = frac(uvScaled);
        [ForceUnroll]
        for (int ch = 0; ch < kNumLatentDims; ch++)
        {
            float code00 = getTexel(tileIndex, mipLevel, ch, uint2(u0, v0));
            float code10 = getTexel(tileIndex, mipLevel, ch, uint2(u1, v0));
            float code01 = getTexel(tileIndex, mipLevel, ch, uint2(u0, v1));
            float code11 = getTexel(tileIndex, mipLevel, ch, uint2(u1, v1));

            code[ch] = lerp(lerp(code00, code10, uvFrac[0]), lerp(code01, code11, uvFrac[0]), uvFrac[1]);
        }
    }

    [BackwardDifferentiable]
    void getCodeBilinearStochastic(float2 uv, float mipLevel, float sample, out float code[kNumLatentDims])
    {
        uint tileIndex = uv2index(uv);
        uv = frac(uv);

        uint mip = sample > frac(mipLevel) ? uint(mipLevel) : uint(mipLevel) + 1;

        uint width = kWidth >> mip;
        uint height = kHeight >> mip;
        float2 uvScaled = uv * float2(width, height) - float2(0.5f, 0.5f);

        uint u0 = clamp(uint(uvScaled[0]), 0, width - 1);
        uint u1 = u0 < width - 1 ? u0 + 1 : 0;
        uint v0 = clamp(uint(uvScaled[1]), 0, height - 1);
        uint v1 = v0 < height - 1 ? v0 + 1 : 0;

        float2 uvFrac = frac(uvScaled);
        [ForceUnroll]
        for (int ch = 0; ch < kNumLatentDims; ch++)
        {
            float code00 = getTexel(tileIndex, mip, ch, uint2(u0, v0));
            float code10 = getTexel(tileIndex, mip, ch, uint2(u1, v0));
            float code01 = getTexel(tileIndex, mip, ch, uint2(u0, v1));
            float code11 = getTexel(tileIndex, mip, ch, uint2(u1, v1));

            code[ch] = lerp(lerp(code00, code10, uvFrac[0]), lerp(code01, code11, uvFrac[0]), uvFrac[1]);
        }
    }

    float getTexel(uint tile, uint level, uint channel, uint2 texel) { return parameters[tile][level][channel][texel]; }

    [BackwardDerivativeOf(getTexel)]
    void bwd_getTexel(uint tile, uint level, uint channel, uint2 texel, float d_out)
    {
        NvInterlockedAddFp32(derivatives[tile][level][channel], texel, d_out);
    }

    void setTexel(uint tile, uint level, uint channel, uint2 texel, float p) { parameters[tile][level][channel][texel] = p; }

    void getOptimizerState(uint tile, uint level, uint channel, uint2 texel, out float p, out float d, out float2 m)
    {
        p = parameters[tile][level][channel][texel];
        d = derivatives[tile][level][channel][texel];
        m = moments[tile][level][channel][texel];
    }

    void updateOptimizerState(uint tile, uint level, uint channel, uint2 texel, float p, float2 m)
    {
        parameters[tile][level][channel][texel] = p;
        moments[tile][level][channel][texel] = m;
    }
};
