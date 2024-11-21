/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#include "CompositeMode.slangh"

/**
 * Simple composite pass that blends two buffers together.
 */

cbuffer CB
{
    uint2 frameDim;
    float scaleA;
    float scaleB;
}

// Inputs
Texture2D<float4> A;
Texture2D<float4> B;

// Output
#if !defined(OUTPUT_FORMAT)
#error OUTPUT_FORMAT is undefined
#elif OUTPUT_FORMAT == OUTPUT_FORMAT_FLOAT
RWTexture2D<float4> output;
#elif OUTPUT_FORMAT == OUTPUT_FORMAT_UINT
RWTexture2D<uint4> output;
#elif OUTPUT_FORMAT == OUTPUT_FORMAT_SINT
RWTexture2D<int4> output;
#else
#error OUTPUT_FORMAT unknown
#endif

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    const uint2 pixel = dispatchThreadId.xy;
    if (any(pixel >= frameDim))
        return;

    float4 result = float4(0.f);
#if !defined(COMPOSITE_MODE)
#error COMPOSITE_MODE is undefined
#elif COMPOSITE_MODE == COMPOSITE_MODE_ADD
    result = (scaleA * A[pixel]) + (scaleB * B[pixel]);
#elif COMPOSITE_MODE == COMPOSITE_MODE_MULTIPLY
    result = (scaleA * A[pixel]) * (scaleB * B[pixel]);
#else
#error COMPOSITE_MODE unknown
#endif

#if OUTPUT_FORMAT != OUTPUT_FORMAT_FLOAT
    result = round(result);
#endif
    output[pixel] = result;
}
