/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/

/**
 * Implements the alias method for sampling from a discrete probability distribution.
 */
struct AliasTable
{
    struct Item
    {
        uint threshold;
        uint indexA;
        uint indexB;
        uint _pad;

        float getThreshold() { return asfloat(threshold); }
        uint getIndexA() { return indexA; }
        uint getIndexB() { return indexB; }
    };

    StructuredBuffer<Item> items;    ///< List of items used for sampling.
    StructuredBuffer<float> weights; ///< List of original weights.
    uint count;                      ///< Total number of weights in the table.
    float weightSum;                 ///< Total sum of all weights in the table.

    /**
     * Sample from the table proportional to the weights.
     * @param[in] index Uniform random index in [0..count).
     * @param[in] rnd Uniform random number in [0..1).
     * @return Returns the sampled item index.
     */
    uint sample(uint index, float rnd)
    {
        Item item = items[index];
        return rnd >= item.getThreshold() ? item.getIndexA() : item.getIndexB();
    }

    /**
     * Sample from the table proportional to the weights.
     * @param[in] rnd Two uniform random number in [0..1).
     * @return Returns the sampled item index.
     */
    uint sample(float2 rnd)
    {
        uint index = min(count - 1, (uint)(rnd.x * count));
        return sample(index, rnd.y);
    }

    /**
     * Get the original weight at a given index.
     * @param[in] index Table index.
     * @return Returns the original weight.
     */
    float getWeight(uint index) { return weights[index]; }
};
