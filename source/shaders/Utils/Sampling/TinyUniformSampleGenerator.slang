/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
__exported import Utils.Sampling.SampleGeneratorInterface;
import Utils.Sampling.Pseudorandom.LCG;
import Utils.Math.HashUtils;
import Utils.Math.BitTricks;

/**
 * Tiny uniform random sample generator.
 *
 * This generator has only 32 bit state and sub-optimal statistical properties.
 * Do not use for anything critical; correlation artifacts may be prevalent.
 */
export struct TinyUniformSampleGenerator : ISampleGenerator
{
    struct Padded
    {
        TinyUniformSampleGenerator internal;
        uint3 _pad;
    };

    /**
     * Initializes the sample generator.
     * @param[in] seed Seed value.
     */
    __init(uint seed) { this.rng = createLCG(seed); }

    /**
     * Initializes the sample generator for a given pixel and sample number.
     * @param[in] pixel Pixel id.
     * @param[in] sampleNumber Sample number.
     */
    __init(uint2 pixel, uint sampleNumber)
    {
        // Use block cipher to generate a pseudorandom initial seed.
        uint seed = blockCipherTEA(interleave_32bit(pixel), sampleNumber).x;
        this.rng = createLCG(seed);
    }

    /**
     * Returns the next sample value. This function updates the state.
     */
    [mutating]
    uint next() { return nextRandom(rng); }

    LCG rng; ///< Simple LCG 32-bit pseudorandom number generator.
};
