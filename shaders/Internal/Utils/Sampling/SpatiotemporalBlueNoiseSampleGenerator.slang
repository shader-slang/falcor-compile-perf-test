/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
__exported import Utils.Sampling.SampleGeneratorInterface;
import Utils.Sampling.Pseudorandom.Xorshift32;
import Utils.Math.BitTricks;
import Utils.Math.HashUtils;

struct SpatiotemporalBlueNoiseSampleGeneratorData
{
    Texture2D<float> textures[64];
    uint frameIndex;
};

ParameterBlock<SpatiotemporalBlueNoiseSampleGeneratorData> gSpatiotemporalBlueNoiseSampleGeneratorData;

float2 R2(uint index)
{
    // Generalized golden ratio to 2d.
    // Solution to x^3 = x + 1
    // AKA plastic constant.
    // from http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    float g = 1.32471795724474602596f;
    return frac(float2(float(index) / g, float(index) / (g * g)));
}

struct SpatiotemporalBlueNoiseSampleGenerator : ISampleGenerator
{
    struct Padded
    {
        SpatiotemporalBlueNoiseSampleGenerator internal;
        uint pad;
    };

    /**
     * Initializes the sample generator for a given pixel and sample number.
     * @param[in] pixel Pixel id.
     * @param[in] sampleNumber Sample number.
     */
    __init(uint2 pixel, uint sampleNumber)
    {
        this.pixel = pixel;
        this.index = sampleNumber;
    }

    /**
     * Returns the next sample value. This function updates the state.
     */
    [mutating]
    uint next()
    {
        uint2 offset = uint2(R2(index) * float2(128.0f, 128.0f));
        index = index + 1;

        let data = gSpatiotemporalBlueNoiseSampleGeneratorData;
        float randF = data.textures[data.frameIndex % 64][(pixel + offset) % uint2(128, 128)];
        return uint(randF / 0x1p-24) << 8;
    }

    uint2 pixel;
    uint index;
};
