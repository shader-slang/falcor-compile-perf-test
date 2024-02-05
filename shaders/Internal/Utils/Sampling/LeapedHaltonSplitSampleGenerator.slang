/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
#include "Internal/Utils/Sampling/SplitSampleGeneratorType.slangh"

__exported import Utils.Sampling.SampleGeneratorInterface;
import Utils.Sampling.Pseudorandom.Xorshift32;
import Utils.Sampling.SampleGenerator;
import Internal.Utils.Sampling.LowDiscrepancy.HaltonSequence;
import Utils.Math.HashUtils;
import Utils.Debug.PixelDebug;

// Precision configuration.
#if SPLIT_SAMPLE_GENERATOR_PRECISION == SPLIT_SAMPLE_GENERATOR_PRECISION_FULL
typedef uint SplitSampleGeneratorCacheType;
#define split_sample_generator_to_cache(x) (x)
#define split_sample_generator_from_cache(x) (x)
#elif SPLIT_SAMPLE_GENERATOR_PRECISION == SPLIT_SAMPLE_GENERATOR_PRECISION_COMPRESSED
typedef uint16_t SplitSampleGeneratorCacheType;
#define split_sample_generator_to_cache(x) (uint16_t)((x) >> 16)
#define split_sample_generator_from_cache(x) ((uint)(x) << 16)
#else
#error Unknown SPLIT_SAMPLE_GENERATOR_PRECISION
#endif

/**
 * Splits a random vector from a SampleGenerator into multiple well-distributed quasirandom vectors.
 * Construct with split1D, split2D, split3D, split4D, ..., or splitND.
 *
 * Query the dimensions of the current quasirandom vector with repeated calls to next() (or the sampleNext*D wrappers).
 * Proceed to the next quasirandom vector with nextInstance().
 *
 * Cranley-Patterson rotates a variant of the Halton sequence with the original pseudorandom numbers.
 * If out of dimensions, uses pseudorandom to extend the rotation vector, although relying on this is not recommended.
 *
 * @param[in] DimensionCount Number of dimensions from the SampleGenerator to split into new random vectors.
 */
struct LeapedHaltonSplitSampleGenerator<let DimensionCount : int> : ISampleGenerator
{
    static const uint kDimensionCount = LeapedHaltonSequence::kDimensionCount; ///< Number of dimensions supported. Upper limit for the
                                                                               ///< DimensionCount parameter.
    static const uint kInstanceCount = LeapedHaltonSequence::kInstanceCount; ///< Number of instances supported. Upper limit for splitting.

    [mutating]
    uint next()
    {
        uint instance, dimension;
        lds.getInstanceAndDimension(instance, dimension);

        // TODO: Assert dimension < DimensionCount and instance < kInstanceCount;

        // Cranley-Patterson rotate the element from the Halton sequence with the values from the feeding RNG.
        uint currentRotation = split_sample_generator_from_cache(rotation[dimension % DimensionCount]);
        uint currentLdsValue = lds.next();

        // assert(dimension < DimensionCount); // TODO: Warn once?
        if (dimension >= DimensionCount)
        {
            // Extend the rotation vector with pseudorandom.
            var rng = Xorshift32(currentRotation ^ jenkinsHash(dimension));
            currentRotation += rng.next();
        }

        return currentLdsValue + currentRotation;
    }

    /**
     * Moves to the next sample vector.
     */
    [mutating]
    void nextInstance()
    {
        // Increment instance and reset dimension.
        uint instance, dimension;
        lds.getInstanceAndDimension(instance, dimension);
        lds.setInstanceAndDimension(instance + 1, 0);
    }

    SplitSampleGeneratorCacheType rotation[DimensionCount]; ///< The random Cranley-Pattersion rotation from the feeding RNG.
    LeapedHaltonSequence lds;                               ///< The deterministic sequence to be Cranley-Patterson rotated.
};

/**
 * Constructs a HaltonSplitSampleGenerator. Use splitND in user code.
 */
LeapedHaltonSplitSampleGenerator<DimensionCount> createSplitSampleGenerator<let DimensionCount : int, S : ISampleGenerator>(inout S sg)
{
    // Store the next random numbers from sg as a rotation vector.
    LeapedHaltonSplitSampleGenerator<DimensionCount> ssg;
    ssg.lds = LeapedHaltonSequence(0);

    for (int i = 0; i < DimensionCount; ++i)
    {
        ssg.rotation[i] = split_sample_generator_to_cache(sg.next());
    }

    return ssg;
}
