#include "Utils/NVAPI.slangh" // For atomics
#include "Utils/Math/MathConstants.slangh"
import Utils.Sampling.SampleGenerator;

struct LatentDictionary
{
    static const uint32_t kNumInstances = NUM_INSTANCES;
    static const uint32_t kNumLevels = NUM_LATENT_DICTIONARY_LEVELS;
    static const uint32_t kNumLatentDims = NUM_LATENT_DICTIONARY_DIMS;
    static const uint32_t kNumLatentDimsPerLevel = NUM_LATENT_DICTIONARY_DIMS_PER_LEVEL;

    RWTexture2D<float> parameters[kNumInstances][kNumLevels][kNumLatentDimsPerLevel];
    RWTexture2D<float> derivatives[kNumInstances][kNumLevels][kNumLatentDimsPerLevel];
    RWTexture2D<float2> moments[kNumInstances][kNumLevels][kNumLatentDimsPerLevel];

    const uint2 sizes[kNumLevels];

    uint2 getSize(uint level) { return sizes[level]; }

    [BackwardDifferentiable]
    float2 cartesian2unit(float3 dir)
    {
        float mu = max(dir.z, 0.f);
        float phi = (atan2(dir.y, dir.x) + M_PI) / (2.f * M_PI);
        return float2(phi, mu);
    }

    [BackwardDifferentiable]
    void getCodeBilinear(float3 dir, out float code[kNumLatentDims], uint instance)
    {
        float2 uv = cartesian2unit(dir);
        [ForceUnroll]
        for (uint level = 0; level < kNumLevels; level++)
        {
            float2 uvScaled = uv * sizes[level];
            uint u0 = min(uint(uvScaled.x), sizes[level].x - 1);
            uint u1 = (u0 + 1) % sizes[level].x;
            uint v0 = min(uint(uvScaled.y), sizes[level].y - 1);
            uint v1 = min(v0 + 1, sizes[level].y - 1);

            float2 uvFrac = frac(uvScaled);
            [ForceUnroll]
            for (int ch = 0; ch < kNumLatentDimsPerLevel; ch++)
            {
                float code00 = getParam(instance, level, ch, uint2(u0, v0));
                float code10 = getParam(instance, level, ch, uint2(u1, v0));
                float code01 = getParam(instance, level, ch, uint2(u0, v1));
                float code11 = getParam(instance, level, ch, uint2(u1, v1));

                code[level * kNumLatentDimsPerLevel + ch] =
                    lerp(lerp(code00, code10, uvFrac[0]), lerp(code01, code11, uvFrac[0]), uvFrac[1]);
            }
        }
    }

    float getParam(uint instance, uint level, uint channel, uint2 texel) { return parameters[instance][level][channel][texel]; }

    [BackwardDerivativeOf(getParam)]
    void bwd_getParam(uint instance, uint level, uint channel, uint2 texel, float d_out)
    {
        NvInterlockedAddFp32(derivatives[instance][level][channel], texel, d_out);
    }

    void getOptimizerState(uint instance, uint level, uint channel, uint2 texel, out float p, out float d, out float2 m)
    {
        p = parameters[instance][level][channel][texel];
        d = derivatives[instance][level][channel][texel];
        m = moments[instance][level][channel][texel];
    }

    void updateOptimizerState(uint instance, uint level, uint channel, uint2 texel, float p, float2 m)
    {
        parameters[instance][level][channel][texel] = p;
        moments[instance][level][channel][texel] = m;
    }
}
