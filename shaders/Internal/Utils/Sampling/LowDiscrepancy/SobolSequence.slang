/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/

/**
 * Implementation of the Sobol' sequence.
 *
 * The Sobol sequence is as an infinite sequence {s_i}_{i=0,1,2,...} where
 * each instance s_i, is an infinite dimensional sample s_i =
 * (s_ij)_{j=0,1,2,...} = (s_(i,0), s_(i,1), s_(i,2), ...).
 * See https://en.wikipedia.org/wiki/Sobol_sequence.
 *
 * Note: This code should NOT be released publicly due to NVIDIA IP.
 */

// Sobol direction numbers. See external/packman/sobol/README.txt.
StructuredBuffer<uint> gSobolMatrix;

/**
 * Low-discrepancy Sobol' sequence generator.
 *
 * This generator has only 32 bit state. It generates samples from one instance
 * of the Sobol' sequence. Each call to next() returns the next dimension.
 */
struct SobolSequence
{
    static const uint kLogDimensions = 7;
    static const uint kDimensions = 1 << kLogDimensions;

    static const uint kDimensionBits = kLogDimensions;
    static const uint kInstanceBits = 32 - kDimensionBits;

    /**
     * Initializes one instance of the Sobol sequence.
     * @param[in] instance Which instance s_i to create.
     */
    __init(uint instance) { this.packInstanceAndDimension(instance, 0); }

    /**
     * Returns the next dimension of the infinite dimensional sample s_i, starting at j=0.
     */
    [mutating]
    uint next()
    {
        uint dimension, i;
        unpackInstanceAndDimension(i, dimension);

        // Because the dimension is only given |kLogDimensions| bits in the
        // SobolSampleGenerator, it automatically wraps around back to the first
        // dimension after the last set of matrices are used, without any
        // additional masking here.
        uint sample = 0;
        for (uint m = dimension * 32; i != 0; i >>= 1, ++m)
        {
            sample ^= gSobolMatrix[m] * (i & 1);
        }

        // Advance to the next dimension.
        // We can increment the dimension by just adding a one bit at its
        // lowest bit.  This works out since the dimension is stored in the
        // high bits of |instanceAndDimension|; when |dimension| equals
        // |kDimensions| - 1 and the dimension is incremented, then the ones
        // carry out of the top of the uint and we're back to dimension 0.
        instanceAndDimension += (1 << kInstanceBits);

        return sample;
    }

    void unpackInstanceAndDimension(out uint instance, out uint dimension)
    {
        instance = instanceAndDimension & ((1 << kInstanceBits) - 1);
        dimension = instanceAndDimension >> kInstanceBits;
    }

    [mutating]
    void packInstanceAndDimension(uint instance, uint dimension)
    {
        // In general, we don't expect the instance to become super large and
        // so if we wanted to live dangerously, we could probably skip the
        // AND. That doesn't seem worth the risks, however...
        instance &= (1 << kInstanceBits) - 1;
        instanceAndDimension = (dimension << kInstanceBits) | instance;
    }

    // We store the current dimension and sample instance packed together
    // into a single uint, where the low |kInstanceBits| hold the instance
    // and the high |kDimensionBits| hold the dimension. (It's worth saving
    // space when SampleGenerators are used in ray payloads.)
    uint instanceAndDimension;
};
