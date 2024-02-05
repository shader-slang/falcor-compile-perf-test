/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
StructuredBuffer<uint> gSequenceData;

/**
 * A sequence of precomputed Sobol' elements.
 *
 * Supports instances from 0 to 255 and dimensions from 0 to 31.
 *
 * The first 256 Sobol' elements are of form 0xXY000000u which allows storing them as single bytes.
 * The numbers are stored in a uint32 array.
 */
struct PrecomputedSobolSequence
{
    /// This must be large enough to store the range [0, kDimensionCount].
    static const uint kDimensionBits = 6;
    /// This must be large enough to store the range [0, kInstanceCount] but should be as large as possible.
    static const uint kInstanceBits = 32 - kDimensionBits;

    /// Fixed by the precomputed look-up table.
    static const uint kDimensionCount = 32;
    /// Theoretical maximum for Sobol elements of form 0xXY000000u.
    static const uint kInstanceCount = 256;

    /// Number of uints per dimension.
    static const uint kDataStride = kDimensionCount / 4;

    /**
     * Initialize one instance of the sequence.
     * @param[in] instance Which instance s_i to create. The sequence repeats every kInstanceCount instances.
     */
    __init(uint instance) { this.instanceAndDimension = instance; }

    /**
     * Returns the next dimension of the sample vector.
     *
     * At most kDimensionCount dimensions are supported.
     */
    [mutating]
    uint next()
    {
        uint instance, dimension;
        getInstanceAndDimension(instance, dimension);

        // Repeat every kInstanceCount instances.
        instance &= kInstanceCount - 1;

        // TODO: Assert dimension < kDimensionCount.
        if (dimension >= kDimensionCount)
            return 0;

        // Index to the currently stored data.
        uint byteIndex = dimension % 4;

        // Read a new block of data for the next four dimensions if needed.
        // Storing was faster than re-reading the bytes every time.
        if (byteIndex == 0)
        {
            nextData = gSequenceData[kDataStride * instance + dimension / 4];
        }

        // Use the next byte from the stored data.
        uint value = (nextData >> (8 * byteIndex)) & 0xFFu;
        value = value << 24;

        incrementDimension();
        return value;
    }

    [mutating]
    void incrementDimension()
    {
        // Increment the dimension by just adding one to its lowest bit. This is not supposed to overflow.
        instanceAndDimension += (1 << kInstanceBits);
    }

    void getInstanceAndDimension(out uint instance, out uint dimension)
    {
        instance = instanceAndDimension & ((1 << kInstanceBits) - 1);
        dimension = instanceAndDimension >> kInstanceBits;
    }

    [mutating]
    void setInstanceAndDimension(uint instance, uint dimension)
    {
        instance &= (1 << kInstanceBits) - 1;
        instanceAndDimension = (dimension << kInstanceBits) | instance;

        // TODO: Update only when needed?
        nextData = gSequenceData[kDataStride * instance + dimension / 4];
    }

    // We store the current dimension and sample instance packed together
    // into a single uint, where the low |kInstanceBits| hold the instance
    // and the high |kDimensionBits| hold the dimension. (It's worth saving
    // space when SampleGenerators are used in ray payloads.)
    uint instanceAndDimension;

    uint nextData; ///< Stored data for the next four dimensions.
};
