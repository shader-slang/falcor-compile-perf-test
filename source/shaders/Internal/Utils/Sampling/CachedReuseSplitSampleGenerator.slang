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
import Utils.Math.HashUtils;

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
 * Splits a random vector from a SampleGenerator by caching and replaying the same DimensionCount dimensions over and over.
 * Construct with split1D, split2D, split3D, split4D, ..., or splitND.
 *
 * Only suitable when splitting between completely different strategies,
 * e.g. between light sampling with one sample and BSDF sampling.
 *
 * Query the dimensions of the current quasirandom vector with repeated calls to next() (or the sampleNext*D wrappers).
 * Proceed to the next quasirandom vector with nextInstance().
 *
 * If out of dimensions, uses pseudorandom to extend the rotation vector although this is not recommended.
 *
 * @param[in] DimensionCount Number of dimensions from the SampleGenerator to split into new random vectors.
 */
struct CachedReuseSplitSampleGenerator<let DimensionCount : int> : ISampleGenerator
{
    static const uint kDimensionCount = (uint)(-1); ///< Number of dimensions supported. No limit.
    static const uint kInstanceCount = (uint)(-1);  ///< Upper limit for splitting. No limit.

    [mutating]
    uint next()
    {
        uint result = split_sample_generator_from_cache(rotation[currentDimension % DimensionCount]);

        // assert(currentDimension < DimensionCount);
        if (currentDimension >= DimensionCount)
        {
            // Note: Repeat the same sequence for all instances as per the promise of reuse.
            var rng = Xorshift32(result ^ jenkinsHash(currentDimension));
            result = rng.next();
        }

        ++currentDimension;
        return result;
    }

    /**
     * Begins the next sample vector.
     */
    [mutating]
    void nextInstance() { currentDimension = 0; }

    SplitSampleGeneratorCacheType rotation[DimensionCount]; ///< The random Cranley-Pattersion rotation from the feeding RNG.
    uint currentDimension;                                  ///< Index of the current dimension.
};

/**
 * Constructs a CachedReuseSplitSmapleGenerator. Use splitND in user code.
 */
CachedReuseSplitSampleGenerator<DimensionCount> createSplitSampleGenerator<let DimensionCount : int, S : ISampleGenerator>(inout S sg)
{
    // Store the next random numbers from sg as a rotation vector.
    CachedReuseSplitSampleGenerator<DimensionCount> ssg;

    for (int i = 0; i < DimensionCount; ++i)
    {
        ssg.rotation[i] = split_sample_generator_to_cache(sg.next());
    }
    ssg.currentDimension = 0;

    return ssg;
}
