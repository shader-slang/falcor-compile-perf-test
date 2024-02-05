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
import Utils.Sampling.Pseudorandom.Xorshift32;
import Internal.Utils.Sampling.LowDiscrepancy.HaltonSequence;
import Utils.Math.BitTricks;
import Utils.Math.HashUtils;

/**
 * Implementation of a SampleGenerator for the Leaped Halton Sequence
 * with a separate leap factor for each dimension.
 *
 * This code requires Shader Model 6.2 due to the Halton Sequence implementation.
 */

/**
 * Generates a variant of the Faure-Lemieux Halton low-discrepancy sequence introduced in
 * the paper "Generalized Halton sequences in 2008: A comparative study".
 *
 * This generator is designed to use little memory bandwidth while still having reasonable
 * low-discrepancy properties.
 *
 * The FL-Halton sequence is variant of the Halton sequence in which the Halton digits are
 * multiplied by well-chosen constants (mod b) in the digit inversion process. This
 * variant instead multiplies the sample index with the same dimension-specific constant,
 * as the sequences formed by leaping are very similar to those formed by scrambling.
 * This does not seem to visibly affect integration performance but allows a more efficient
 * implementation.
 *
 * The sequence is transformed by Cranley-Patterson rotation which is different for every
 * pixel and changes every time the Halton sequence repeats. If more dimensions are queried than
 * are provided, falls back to uniform random.
 */
struct LeapedHaltonSampleGenerator : ISampleGenerator
{
    struct Padded
    {
        LeapedHaltonSampleGenerator internal;
        uint2 _pad;
    };

    static const uint kDimensionCount = LeapedHaltonSequence::kDimensionCount;
    static const uint kInstanceCount = min(4096, LeapedHaltonSequence::kInstanceCount); // Limit index to make Halton faster to evaluate.
                                                                                        // Does not make a big difference.

    /**
     * Initializes the sample generator for a given pixel and sample instance.
     * @param[in] pixel Pixel id.
     * @param[in] instance Sample instance.
     */
    __init(uint2 pixel, uint instance)
    {
        this.lds = LeapedHaltonSequence(instance % kInstanceCount);

        // Change per-pixel Cranley-Patterson rotation every kInstanceCount samples.
        var seed = jenkinsHash(interleave_32bit(pixel) ^ jenkinsHash(instance / kInstanceCount));
        this.rng = Xorshift32(seed);
    }

    /**
     * Returns the next sample value. This function updates the state.
     */
    [mutating]
    uint next()
    {
        // Generate scrambled Halton samples for the first kDimensionCount dimensions, then fall back to uniform random.
        uint result;

        uint instance, dimension;
        lds.getInstanceAndDimension(instance, dimension);

        if (dimension < kDimensionCount)
        {
            // Return Halton numbers.
            uint rotation = rng.next();
            result = rotation + lds.next();
            lds.incrementDimension();
        }
        else
        {
            if (dimension == kDimensionCount)
            {
                // Reconfigure as a uniform sampler. From this point on rng depends also on the instance.

                // Scramble by the hashed instance number to create a unique stream.
                rng = Xorshift32(rng.state ^ jenkinsHash(instance));

                // Increment one last time to avoid further reconfigurations.
                lds.incrementDimension();
            }

            // Return uniform random.
            result = rng.next();
        }

        return result;
    }

    // TODO: Consider a union after Slang gets support for them, as only one of these is needed at a time.
    //       We would need a 31-bit RNG for that though.

    Xorshift32 rng;
    LeapedHaltonSequence lds;
};
