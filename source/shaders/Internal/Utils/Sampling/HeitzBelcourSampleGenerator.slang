/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
__exported import Utils.Sampling.SampleGeneratorInterface;
import Utils.Sampling.Pseudorandom.LCG;

struct HeitzBelcourSampleGeneratorData
{
    Texture2D<uint> seeds;
};

ParameterBlock<HeitzBelcourSampleGeneratorData> gHeitzBelcourSampleGeneratorData;

struct HeitzBelcourSampleGenerator : ISampleGenerator
{
    struct Padded
    {
        HeitzBelcourSampleGenerator internal;
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
        let seeds = gHeitzBelcourSampleGeneratorData.seeds;
        uint2 seedsDim;
        seeds.GetDimensions(seedsDim.x, seedsDim.y);
        pixel = pixel % seedsDim;
        uint seed = seeds[pixel];
        return HeitzBelcourSampleGenerator(seed);
    }

    /**
     * Returns the next sample value. This function updates the state.
     */
    [mutating]
    uint next() { return nextRandom(rng); }

    LCG rng; ///< Simple LCG 32-bit pseudorandom number generator.
};
