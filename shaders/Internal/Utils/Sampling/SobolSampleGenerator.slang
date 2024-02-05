/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
__exported import Utils.Sampling.SampleGeneratorInterface;
import Utils.Sampling.Pseudorandom.LCG;
import Internal.Utils.Sampling.Randomization;
import Internal.Utils.Sampling.LowDiscrepancy.SobolSequence;
import Utils.Math.BitTricks;
import Utils.Math.HashUtils;

/**
 * Low-discrepancy Sobol' sequence sample generator.
 *
 * This generator has only 64 bit state. It generates samples from the Sobol'
 * sequence and applies per-pixel randomization to decorrelate the generated
 * sequences.
 */
struct SobolSampleGenerator : ISampleGenerator
{
    struct Padded
    {
        SobolSampleGenerator internal;
        uint2 _pad;
    };

    /**
     * Initializes the sample generator for a given pixel and sample number.
     * @param[in] pixel Pixel id.
     * @param[in] sampleNumber Sample number.
     */
    __init(uint2 pixel, uint sampleNumber)
    {
        // Create Sobol sequence with 'sampleNumber' as instance index.
        this.sequence = SobolSequence(sampleNumber);

        // Create random digit scrambler initialized by a per-pixel hash.
        // Note: the hash must not depend on 'sampleNumber'; it has to be the same
        // for all samples in a pixel. The hash must have high entropy.
        // We currently use a block cipher to generate a pseudorandom seed for
        // an LCG pseudorandom number generator, which computes the hash.
        uint seed = blockCipherTEA(interleave_32bit(pixel), 0xbeeff00d).x;
        LCG rng = createLCG(seed);
        uint h = nextRandom(rng);
        this.scrambler = createKolligKellerScrambler(h);
    }

    /**
     * Returns the next sample value. This function updates the state.
     */
    [mutating]
    uint next()
    {
        uint sample = sequence.next();
        sample = randomizeSample(scrambler, sample);
        return sample;
    }

    SobolSequence sequence;          ///< Sobol sequence generator.
    KolligKellerScrambler scrambler; ///< Randomization method (currently Kollig-Keller scrambling).
};
