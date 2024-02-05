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

/**
 * The host sets the SPLIT_SAMPLE_GENERATOR_TYPE define to select the split sample generator.
 *
 * This code typedefs the chosen type to the type 'SplitSampleGenerator'.
 */

// Import the chosen implementation.
#if SPLIT_SAMPLE_GENERATOR_TYPE == SPLIT_SAMPLE_GENERATOR_TYPE_LEAPED_HALTON
import Internal.Utils.Sampling.LeapedHaltonSplitSampleGenerator;

// TODO: Replace with a generic typedef once available in Slang.
struct SplitSampleGenerator<let DimensionCount : int>
{
    typedef LeapedHaltonSplitSampleGenerator<DimensionCount> type;
};
#elif SPLIT_SAMPLE_GENERATOR_TYPE == SPLIT_SAMPLE_GENERATOR_TYPE_CACHED_REUSE
import Internal.Utils.Sampling.CachedReuseSplitSampleGenerator;

// TODO: Replace with a generic typedef once available in Slang.
struct SplitSampleGenerator<let DimensionCount : int>
{
    typedef CachedReuseSplitSampleGenerator<DimensionCount> type;
};
#else
#error Unsupported SPLIT_SAMPLE_GENERATOR_TYPE
#endif

// The main interface for using SplitSampleGenerators.

/**
 * Note: splitND will become more convenient when automatic generic parameter detection has been implemented to Slang,
 * as then e.g. splitND<6>(...) will work instead of splitND<6, SampleGeneratorType>(...).
 *
 * The convenience wrappers below are quite useful until that.
 */

/**
 * Splits a SampleGenerator into multiple quasirandom samples.
 * @param[in] DimensionCount Number of dimensions from the SampleGenerator to split into new random vectors.
 * @param[in] sg The SampleGenerator to split. See the details from the used SplitSampleGenerator.
 */
SplitSampleGenerator<DimensionCount>::type splitND<let DimensionCount : int, S : ISampleGenerator>(inout S sg)
{
    // TODO: Assert 1 <= DimensionCount < SplitSampleGenerator<DimensionCount>::kDimensionCount
    return createSplitSampleGenerator<DimensionCount, S>(sg);
}

/**
 * Splits a SampleGenerator into multiple quasirandom 1D samples.
 * @see splitND
 */
SplitSampleGenerator<1>::type split1D<S : ISampleGenerator>(S sg)
{
    return splitND<1, S>(sg);
}

/**
 * Splits a SampleGenerator into multiple quasirandom 2D samples.
 * @see splitND
 */
SplitSampleGenerator<2>::type split2D<S : ISampleGenerator>(inout S sg)
{
    return splitND<2, S>(sg);
}

/**
 * Splits a SampleGenerator into multiple quasirandom 3D samples.
 * @see splitND
 */
SplitSampleGenerator<3>::type split3D<S : ISampleGenerator>(inout S sg)
{
    return splitND<3, S>(sg);
}

/**
 * Splits a SampleGenerator into multiple quasirandom 4D samples.
 * @see splitND
 */
SplitSampleGenerator<4>::type split4D<S : ISampleGenerator>(inout S sg)
{
    return splitND<4, S>(sg);
}

/**
 * Splits a SampleGenerator into multiple quasirandom 5D samples.
 * @see splitND
 */
SplitSampleGenerator<5>::type split5D<S : ISampleGenerator>(inout S sg)
{
    return splitND<5, S>(sg);
}

/**
 * Splits a SampleGenerator into multiple quasirandom 6D samples.
 * @see splitND
 */
SplitSampleGenerator<6>::type split6D<S : ISampleGenerator>(inout S sg)
{
    return splitND<6, S>(sg);
}
