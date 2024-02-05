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

struct LatentHashGrid
{
    static const uint32_t kNumLevels = NUM_HASHGRID_LEVELS;
    static const uint32_t kNumLatentDims = NUM_LATENT_DIMS;
    static const uint32_t kNumLatentDimsPerLevel = NUM_LATENT_DIMS_PER_LEVEL;

    RWTexture2D<float> parameters[kNumLevels][kNumLatentDimsPerLevel];
    RWTexture2D<float> derivatives[kNumLevels][kNumLatentDimsPerLevel];
    RWTexture2D<float2> moments[kNumLevels][kNumLatentDimsPerLevel];

    static const uint kPi1 = 1;
    static const uint kPi2 = 2654435761;
    const uint resolutions[kNumLevels]; // Grid resolutions at the individual levels of the hierarchy.
    const uint2 sizes[kNumLevels];      // Buffer sizes at the individual levels of the hierarchy.

    uint2 getBufferSize(uint level) { return sizes[level]; }

    uint2 getDenseIndex(uint u, uint v, uint level) { return uint2(u, v); }

    uint2 getHashedIndex(uint u, uint v, uint level)
    {
        uint index = ((u * kPi1) ^ (v * kPi2)) % (sizes[level].x * sizes[level].y);
        return uint2(index % sizes[level].x, index / sizes[level].x);
    }

    [BackwardDifferentiable]
    void getCodeBilinear(float2 uv, out float code[kNumLatentDims])
    {
        uv = frac(uv);

        uint2 index00, index10, index01, index11;
        for (uint level = 0; level < kNumLevels; level++)
        {
            float2 uvScaled = uv * resolutions[level];
            uint u0 = uint(floor(uvScaled.x));
            uint u1 = min(u0 + 1, resolutions[level] - 1);
            uint v0 = uint(floor(uvScaled.y));
            uint v1 = min(v0 + 1, resolutions[level] - 1);
            uint numGridEntries = resolutions[level] * resolutions[level];
            if (numGridEntries == sizes[level].x * sizes[level].y)
            {
                index00 = getDenseIndex(u0, v0, level);
                index10 = getDenseIndex(u1, v0, level);
                index01 = getDenseIndex(u0, v1, level);
                index11 = getDenseIndex(u1, v1, level);
            }
            else
            {
                index00 = getHashedIndex(u0, v0, level);
                index10 = getHashedIndex(u1, v0, level);
                index01 = getHashedIndex(u0, v1, level);
                index11 = getHashedIndex(u1, v1, level);
            }

            float2 uvFrac = frac(uvScaled);
            [ForceUnroll]
            for (int ch = 0; ch < kNumLatentDimsPerLevel; ch++)
            {
                float code00 = getParam(level, ch, index00);
                float code10 = getParam(level, ch, index10);
                float code01 = getParam(level, ch, index01);
                float code11 = getParam(level, ch, index11);

                code[level * kNumLatentDimsPerLevel + ch] =
                    lerp(lerp(code00, code10, uvFrac[0]), lerp(code01, code11, uvFrac[0]), uvFrac[1]);
            }
        }
    }

    float getParam(uint level, uint channel, uint2 texel) { return parameters[level][channel][texel]; }

    [BackwardDerivativeOf(getParam)]
    void bwd_getParam(uint level, uint channel, uint2 texel, float d_out)
    {
        NvInterlockedAddFp32(derivatives[level][channel], texel, d_out);
    }

    void getOptimizerState(uint level, uint channel, uint2 texel, out float p, out float d, out float2 m)
    {
        p = parameters[level][channel][texel];
        d = derivatives[level][channel][texel];
        m = moments[level][channel][texel];
    }

    void updateOptimizerState(uint level, uint channel, uint2 texel, float p, float2 m)
    {
        parameters[level][channel][texel] = p;
        moments[level][channel][texel] = m;
    }
};
