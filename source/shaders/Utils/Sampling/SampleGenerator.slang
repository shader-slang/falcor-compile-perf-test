/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#include "Utils/Sampling/SampleGeneratorType.slangh"

__exported import Utils.Sampling.SampleGeneratorInterface;

/**
 * The host sets the SAMPLE_GENERATOR_TYPE define to select sample generator.
 *
 * This code typedefs the chosen type to the type 'SampleGenerator'.
 * All sample generators adheres to the same interface, but note that the
 * size of the 'SampleGenerator' type may vary depending on their state size.
 *
 * If SAMPLE_GENERATOR_TYPE is not defined, a compile-time error is printed.
 *
 * The 'SampleGenerator.Padded' type holds a SampleGenerator plus additional
 * padding to make the struct a multiple of 16B.
 */

#if defined(SAMPLE_GENERATOR_TYPE) && SAMPLE_GENERATOR_TYPE == SAMPLE_GENERATOR_TINY_UNIFORM
import Utils.Sampling.TinyUniformSampleGenerator;
typedef TinyUniformSampleGenerator SampleGenerator;
#elif defined(SAMPLE_GENERATOR_TYPE) && SAMPLE_GENERATOR_TYPE == SAMPLE_GENERATOR_UNIFORM
import Utils.Sampling.UniformSampleGenerator;
typedef UniformSampleGenerator SampleGenerator;
#endif
