/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import Utils.Sampling.Pseudorandom.Xoshiro;
import Utils.Sampling.Pseudorandom.SplitMix64;
import Utils.Sampling.Pseudorandom.LCG;

ByteAddressBuffer seed;
RWStructuredBuffer<uint> result;       // Result buffer for 32-bit generators.
RWStructuredBuffer<uint64_t> result64; // Result buffer for 64-bit generators.

#define kInstances 256
#define kDimensions 64

[numthreads(kInstances, 1, 1)]
void testXoshiro(uint3 threadId: SV_DispatchThreadID)
{
    const uint i = threadId.x;

    // Create pseudorandom number generator.
    uint4 state = seed.Load4(i * 16);
    const uint s[4] = { state.x, state.y, state.z, state.w };
    Xoshiro128StarStar rng = createXoshiro128StarStar(s);

    // Generate random numbers.
    for (uint j = 0; j < kDimensions; j++)
    {
        uint val = nextRandom(rng);
        result[j * kInstances + i] = val;
    }
}

[numthreads(kInstances, 1, 1)]
void testSplitMix(uint3 threadId: SV_DispatchThreadID)
{
    const uint i = threadId.x;

    // Create pseudorandom number generator.
    uint2 state = seed.Load2(i * 8);
    SplitMix64 rng = createSplitMix64(state.x, state.y);

    // Generate random numbers.
    for (uint j = 0; j < kDimensions; j++)
    {
        uint64_t val = nextRandom64(rng);
        result64[j * kInstances + i] = val;
    }
}

[numthreads(kInstances, 1, 1)]
void testLCG(uint3 threadId: SV_DispatchThreadID)
{
    const uint i = threadId.x;

    // Create pseudorandom number generator.
    uint state = seed.Load(i * 4);
    LCG rng = createLCG(state);

    // Generate random numbers.
    for (uint j = 0; j < kDimensions; j++)
    {
        uint val = nextRandom(rng);
        result[j * kInstances + i] = val;
    }
}
