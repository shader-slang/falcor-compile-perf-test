/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
__exported import Utils.Sampling.SampleGeneratorInterface;
import Utils.Sampling.Pseudorandom.Xoshiro;
import Utils.Sampling.Pseudorandom.SplitMix64;
import Utils.Math.BitTricks;

/**
 * Default uniform pseudorandom number generator.
 *
 * This generator has 128 bit state and should have acceptable statistical
 * properties for most rendering applications.
 *
 * This sample generator requires shader model 6.0 or above.
 */
export struct UniformSampleGenerator : ISampleGenerator
{
    struct Padded
    {
        UniformSampleGenerator internal;
    }

    /**
     * Initializes the sample generator for a given pixel and sample number.
     * @param[in] pixel Pixel id.
     * @param[in] sampleNumber Sample number.
     */
    __init(uint2 pixel, uint sampleNumber)
    {
        UniformSampleGenerator sampleGenerator;

        // Use SplitMix64 generator to generate a good pseudorandom initial state.
        // The pixel coord is expected to be max 28 bits (16K^2 is the resource limit in D3D12).
        // The sample number is expected to be practically max ~28 bits, e.g. 16spp x 16M samples.
        // As long as both stay <= 32 bits, we will always have a unique initial seed.
        // This is however no guarantee that the generated sequences will never overlap,
        // but it is very unlikely. For example, with these most extreme parameters of
        // 2^56 sequences of length L, the probability of overlap is P(overlap) = L*2^-16.
        SplitMix64 rng = createSplitMix64(interleave_32bit(pixel), sampleNumber);
        uint64_t s0 = nextRandom64(rng);
        uint64_t s1 = nextRandom64(rng);
        uint seed[4] = { uint(s0), uint(s0 >> 32), uint(s1), uint(s1 >> 32) };

        // Create xoshiro128** pseudorandom generator.
        this.rng = createXoshiro128StarStar(seed);
    }

    /**
     * Returns the next sample value. This function updates the state.
     */
    [mutating]
    uint next() { return nextRandom(rng); }

    Xoshiro128StarStar rng;
};
