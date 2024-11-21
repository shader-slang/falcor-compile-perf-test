/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/

/**
 * Slang interface for sample generator implementations.
 */
interface ISampleGenerator
{
    /**
     * Returns the next sample value. This function updates the state.
     */
    [mutating]
    uint next();
};

/**
 * Convenience functions for generating 1D/2D/3D values in the range [0,1).
 *
 * Note: These are global instead of member functions in the sample generator
 * interface, as there seems to be no way in Slang currently to specify default
 * implementations without duplicating the code into all classes that implement
 * the interace.
 */

float sampleNext1D<S : ISampleGenerator>(inout S sg)
{
    // Use upper 24 bits and divide by 2^24 to get a number u in [0,1).
    // In floating-point precision this also ensures that 1.0-u != 0.0.
    uint bits = sg.next();
    return (bits >> 8) * 0x1p-24;
}

float2 sampleNext2D<S : ISampleGenerator>(inout S sg)
{
    float2 sample;
    // Don't use the float2 initializer to ensure consistent order of evaluation.
    sample.x = sampleNext1D(sg);
    sample.y = sampleNext1D(sg);
    return sample;
}

float3 sampleNext3D<S : ISampleGenerator>(inout S sg)
{
    float3 sample;
    // Don't use the float3 initializer to ensure consistent order of evaluation.
    sample.x = sampleNext1D(sg);
    sample.y = sampleNext1D(sg);
    sample.z = sampleNext1D(sg);
    return sample;
}

float4 sampleNext4D<S : ISampleGenerator>(inout S sg)
{
    float4 sample;
    // Don't use the float4 initializer to ensure consistent order of evaluation.
    sample.x = sampleNext1D(sg);
    sample.y = sampleNext1D(sg);
    sample.z = sampleNext1D(sg);
    sample.w = sampleNext1D(sg);
    return sample;
}
