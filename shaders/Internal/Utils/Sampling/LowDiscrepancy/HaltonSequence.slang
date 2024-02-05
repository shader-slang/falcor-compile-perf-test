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
 * Implementation of the Halton sequence and a leaped variation of the Faure-Lemieux Halton sequence.
 *
 * The Halton sequence is an infinite sequence {s_i}_{i=0,1,2,...} where each
 * instance s_i is an infinite-dimensional sample s_i =
 * (s_ij)_j={j=0,1,2,...} = (s_(i,0), s_(i,1) = s_(i,2), ...).
 * See https://en.wikipedia.org/wiki/Halton_sequence.
 *
 * See "Generalized Halton sequences in 2008: A comparative study" (Faure & Lemieux 2008)
 * for the Faure-Lemieux Halton sequence. This implementation uses their scramble multipliers
 * as per-dimension leaps for performance reasons. This does not seem to affect the quality
 * of the sequence at all.
 *
 * This code requires Shader Model 6.2 for the uint16_t array.
 */

/**
 * The Halton implementation and HaltonSequence support all integers that are perfectly representable
 * as single-precision floats, i.e., up to 2^24.
 *
 * They might work quite reasonably for even higher integers but there is no guarantee.
 *
 * For LeapedHaltonSequence, the greatest index times the largest leap (95 or 152) must be at most 2^24 = 16777216.
 */

// Number of supported Halton dimensions. Supported values: 32 and 50.
#define HALTON_BASE_COUNT 32

// Maximum sample count for HaltonSequence. 1u << 24.
#define HALTON_SAMPLE_COUNT 16777216

#if HALTON_BASE_COUNT == 32
// Maximum sample count for LeapedHaltonSequence for up to 32 dimensions. (1u << 24) / 95.
#define LEAPED_HALTON_SAMPLE_COUNT 176602
#elif HALTON_BASE_COUNT == 50
// Maximum sample count for LeapedHaltonSequence for up to 50 dimensions. (1u << 24) / 152.
#define LEAPED_HALTON_SAMPLE_COUNT 110376
#else
#error HALTON_BASE_COUNT is not 32 or 50
#endif

// Storing uint8 values in uint16.
#define halton_pack_leap_base(x, y) (((x) << 8u) | (y))
#define halton_unpack_base(x) ((x)&0xFFu)
#define halton_unpack_leap(x) ((x) >> 8u)

/**
 * Scramble multipliers from Faure & Lemieux. These work equally well as leaps too.
 * Bases 1-32: [1, 1, 3, 3, 4, 9, 7, 5, 9, 18, 18, 8, 13, 31, 9, 19, 36, 33, 21, 44, 43, 61, 60, 56, 26, 71, 32, 77, 26, 95, 92, 47]
 * Bases 33-50: [29, 61, 57, 69, 115, 63, 92, 31, 104, 126, 50, 80, 55, 152, 114, 80, 83, 97]
 */

// clang-format off
// Returns the leap and base by dimension index. Note: Requires Shader Model 6.2.
static const uint16_t kHaltonData[HALTON_BASE_COUNT] = {
    halton_pack_leap_base(1, 2),
    halton_pack_leap_base(1, 3),
    halton_pack_leap_base(3, 5),
    halton_pack_leap_base(3, 7),
    halton_pack_leap_base(4, 11),
    halton_pack_leap_base(9, 13),
    halton_pack_leap_base(7, 17),
    halton_pack_leap_base(5, 19),
    halton_pack_leap_base(9, 23),
    halton_pack_leap_base(18, 29),
    halton_pack_leap_base(18, 31),
    halton_pack_leap_base(8, 37),
    halton_pack_leap_base(13, 41),
    halton_pack_leap_base(31, 43),
    halton_pack_leap_base(9, 47),
    halton_pack_leap_base(19, 53),
    halton_pack_leap_base(36, 59),
    halton_pack_leap_base(33, 61),
    halton_pack_leap_base(21, 67),
    halton_pack_leap_base(44, 71),
    halton_pack_leap_base(43, 73),
    halton_pack_leap_base(61, 79),
    halton_pack_leap_base(60, 83),
    halton_pack_leap_base(56, 89),
    halton_pack_leap_base(26, 97),
    halton_pack_leap_base(71, 101),
    halton_pack_leap_base(32, 103),
    halton_pack_leap_base(77, 107),
    halton_pack_leap_base(26, 109),
    halton_pack_leap_base(95, 113),
    halton_pack_leap_base(92, 127),
    halton_pack_leap_base(47, 131), // 32 bases
#if HALTON_BASE_COUNT == 50
    halton_pack_leap_base(29, 137),
    halton_pack_leap_base(61, 139),
    halton_pack_leap_base(57, 149),
    halton_pack_leap_base(69, 151),
    halton_pack_leap_base(115, 157),
    halton_pack_leap_base(63, 163),
    halton_pack_leap_base(92, 167),
    halton_pack_leap_base(31, 173),
    halton_pack_leap_base(104, 179),
    halton_pack_leap_base(126, 181),
    halton_pack_leap_base(50, 191),
    halton_pack_leap_base(80, 193),
    halton_pack_leap_base(55, 197),
    halton_pack_leap_base(152, 199),
    halton_pack_leap_base(114, 211),
    halton_pack_leap_base(80, 223),
    halton_pack_leap_base(83, 227),
    halton_pack_leap_base(97, 229) // 50 bases
#endif
};
// clang-format on

// Returns the leap and base by dimension index.
void getHaltonLeapAndBase(uint index, out uint leap, out uint base)
{
    uint u = kHaltonData[index];
    leap = halton_unpack_leap(u);
    base = halton_unpack_base(u);
}

// TODO: HaltonSequence and LeapedHaltonSequence share a lot of code with SobolSequence.
///      A common base class might be useful once Slang gets support for inheritance.

/**
 * Low-discrepancy traditional Halton sequence generator.
 *
 * The traditional Halton sequence is not very good for high dimension counts.
 * Consider using LeapedHaltonSequence if you need more than 3 dimensions.
 *
 * Supports kDimensionCount dimensions and kInstanceCount different instances
 * (sample vectors). The 'dimension' and 'instance' variables have enough bits
 * to represent 'kDimensionCount' and 'kInstanceCount' for end-of-sequence
 * detection but the user is responsible that next() is never called with
 * dimension or instance greater than or equal to the above maximums.
 *
 * This generator has only 32 bit state. It generates samples from one instance
 * of the Halton sequence. Each call to next() returns the next dimension.
 */
struct HaltonSequence
{
    static const uint kDimensionBits = 6;                  // This should be large enough to store kDimensionCount.
    static const uint kInstanceBits = 32 - kDimensionBits; // This should be large enough to store kInstanceCount.

    static const uint kDimensionCount = HALTON_BASE_COUNT;
    static const uint kInstanceCount = HALTON_SAMPLE_COUNT;

    /**
     * Initialize one instance of the Halton sequence.
     * @param[in] instance Which instance s_i to create.
     */
    __init(uint instance) { this.setInstanceAndDimension(instance, 0); }

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

        // TODO: Assert dimension < kDimensionCount and instance < kInstanceCount.

        uint leap, base;
        getHaltonLeapAndBase(dimension, leap, base);

        uint result = halton(instance, base);

        incrementDimension();
        return result;
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
    }

    // We store the current dimension and sample instance packed together
    // into a single uint, where the low |kInstanceBits| hold the instance
    // and the high |kDimensionBits| hold the dimension. (It's worth saving
    // space when SampleGenerators are used in ray payloads.)
    uint instanceAndDimension;
};

/**
 * Low-discrepancy Leaped Faure-Lemieux Halton sequence generator.
 *
 * Supports kDimensionCount dimensions and kInstanceCount different instances
 * (sample vectors). The 'dimension' and 'instance' variables have enough bits
 * to represent 'kDimensionCount' and 'kInstanceCount' for end-of-sequence
 * detection but the user is responsible that next() is never called with
 * dimension or instance greater than or equal to the above maximums.
 *
 * This sequence is a variant of the Faure-Lemieux Halton low-discrepancy
 * sequence introduced in the paper "Generalized Halton sequences in 2008: A
 * comparative study".
 *
 * The FL-Halton sequence is variant of the Halton sequence in which the
 * Halton digits are multiplied by well-chosen constants (mod b) in the digit
 * inversion process. This variant instead multiplies the instance index with
 * the same constant, as the sequences formed by leaping are very similar to
 * those formed by scrambling. This does not seem to visibly affect integration
 * performance but allows a more efficient implementation.
 *
 * This generator has only 32 bit state. It generates samples from one instance
 * of the FL-Halton sequence. Each call to next() returns the next dimension.
 */
struct LeapedHaltonSequence
{
    static const uint kDimensionBits = 6;                  // This should be large enough to store kDimensionCount.
    static const uint kInstanceBits = 32 - kDimensionBits; // This should be large enough to store kInstanceCount.

    static const uint kDimensionCount = HALTON_BASE_COUNT;
    static const uint kInstanceCount = LEAPED_HALTON_SAMPLE_COUNT;

    /**
     * Initialize one instance of the Halton sequence.
     * @param[in] instance Which instance s_i to create.
     */
    __init(uint instance) { this.setInstanceAndDimension(instance, 0); }

    /**
     * Returns the next dimension of the infinite-dimensional sample vector.
     *
     * At most kDimensionCount dimensions are supported.
     */
    [mutating]
    uint next()
    {
        uint instance, dimension;
        getInstanceAndDimension(instance, dimension);

        // TODO: Assert dimension < kDimensionCount and instance < kInstanceCount.

        uint leap, base;
        getHaltonLeapAndBase(dimension, leap, base);

        uint result = halton(leap * instance, base);

        incrementDimension();
        return result;
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
    }

    // We store the current dimension and sample instance packed together
    // into a single uint, where the low |kInstanceBits| hold the instance
    // and the high |kDimensionBits| hold the dimension. (It's worth saving
    // space when SampleGenerators are used in ray payloads.)
    uint instanceAndDimension;
};

/**
 * Returns elements of the Halton low-discrepancy sequence.
 *
 * Computes in single-precision floating point and assumes index <= 2^24.
 * Avoids all divisions in the inner loop but requires floor.
 *
 * @param[in] index Index of the queried element. Must be at most 2^24.
 * @param[in] base Base for the digit inversion. Should be the smallest unused prime number.
 */
uint halton(uint index, uint base)
{
    float i = index;
    float result = 0.0;
    float factor = 1.0;
    float onePerBase = 1.0 / base;

    while (i > 0)
    {
        factor *= onePerBase;
        float iNext = floor(i * onePerBase);
        result += factor * (i - iNext * base);
        i = iNext;
    }

    return (uint)(result * 0x1p32);
}
