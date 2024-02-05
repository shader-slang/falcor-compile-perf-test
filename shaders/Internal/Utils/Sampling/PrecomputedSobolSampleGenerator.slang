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
import Internal.Utils.Sampling.LowDiscrepancy.PrecomputedSobolSequence;
import Utils.Math.BitTricks;
import Utils.Math.HashUtils;

struct PrecomputedSobolSampleGenerator : ISampleGenerator
{
    static const uint kDimensionCount = PrecomputedSobolSequence::kDimensionCount;
    static const uint kInstanceCount = PrecomputedSobolSequence::kInstanceCount;

    /**
     * Initializes the sample generator for a given pixel and sample instance.
     * @param[in] pixel Pixel id.
     * @param[in] instance Sample instance.
     */
    __init(uint2 pixel, uint instance)
    {
        this.lds = PrecomputedSobolSequence(instance % kInstanceCount);

        // Change the per-pixel scramble every kInstanceCount samples.
        var seed = jenkinsHash(interleave_32bit(pixel) ^ jenkinsHash(instance / kInstanceCount));
        this.rng = Xorshift32(seed);

        // Note: We could in theory use the KolligKellerScrambler used by SobolSampleGenerator but
        //       we would still need the Xorshift32 generator in case we run out of dimensions.
    }

    /**
     * Returns the next sample value. This function updates the state.
     */
    [mutating]
    uint next()
    {
        // Generate scrambled Sobol samples for the first kDimensionCount dimensions, then fall back to uniform random.
        uint result;

        uint instance, dimension;
        lds.getInstanceAndDimension(instance, dimension);

        if (dimension < kDimensionCount)
        {
            // Return a scrambled version of the precomputed number.
            uint rotation = rng.next();
            result = rotation ^ lds.next();
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
    PrecomputedSobolSequence lds;
};
