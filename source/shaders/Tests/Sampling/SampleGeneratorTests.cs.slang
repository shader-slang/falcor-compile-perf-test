/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import Utils.Sampling.SampleGenerator;

cbuffer CB
{
    uint3 gDispatchDim;
    uint gDimensions;
}

RWStructuredBuffer<float> result;

[numthreads(16, 16, 1)]
void test(uint3 threadId: SV_DispatchThreadID)
{
    if (any(threadId >= gDispatchDim))
        return;

    // Create sample generator.
    SampleGenerator sg = SampleGenerator(threadId.xy, threadId.z);

    // Generate samples.
    // The output is a 4D tensor of samples, stored in memory as Z instances
    // of 2D tiles XY stored in scanline order, where XYZ is the dispatch dimensions.
    // Each element consists of an N-dimensional sample.
    const uint pixelIdx = threadId.y * gDispatchDim.x + threadId.x;
    const uint tileOffset = threadId.z * (gDispatchDim.x * gDispatchDim.y);
    const uint offset = (tileOffset + pixelIdx) * gDimensions;

    for (uint i = 0; i < gDimensions; i += 8)
    {
        float u0 = sampleNext1D(sg);
        float u1 = sampleNext1D(sg);
        float u2 = sampleNext1D(sg);

        result[offset + i + 0] = u0;
        result[offset + i + 1] = u1;
        result[offset + i + 2] = u2;

        float2 v2 = sampleNext2D(sg);

        result[offset + i + 3] = v2.x;
        result[offset + i + 4] = v2.y;

        float3 v3 = sampleNext3D(sg);

        result[offset + i + 5] = v3.x;
        result[offset + i + 6] = v3.y;
        result[offset + i + 7] = v3.z;
    }
}
