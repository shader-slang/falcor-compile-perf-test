/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/

/** Compute pass that writes the pass outputs based on the final path state.

    The dispatch dimensions is one thread per pixel in the output buffers.
    The program iterates over the individual samples to average them.
*/

import RenderPasses.LegacyPathTracer.PathTracer.StaticParams;
import RenderPasses.LegacyPathTracer.PathTracer.PathTracerParams;
import RenderPasses.LegacyPathTracer.WavefrontPathTracer.PathState;

cbuffer PerFrameCB
{
    PathTracerParams gParams;
}

StructuredBuffer<float3> gPathOutput; ///< Accumulated path contribution.
RWTexture2D<float4> gOutputColor;     ///< Final pixel linear color.

// TODO: Profile different thread group sizes. Make it scriptable.
[numthreads(16, 16, 1)]
void writeOutputs(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    const uint2 pixel = dispatchThreadId.xy;
    if (any(pixel >= gParams.frameDim))
        return;

    float3 outColor = float3(0);

    // Compute the final path contribution by averaging the samples.
    for (uint sampleIdx = 0; sampleIdx < kSamplesPerPixel; sampleIdx++)
    {
        const uint idx = getIdx(uint3(pixel, sampleIdx));

        float3 L = gPathOutput[idx];
        if (gParams.clampSamples)
        {
            // Note the comparison is written so that NaNs propagate (unless the compiler rewrites it).
            // TODO: Check the generated code that this is the case.
            L = select(L > gParams.clampThreshold, gParams.clampThreshold, L);
        }

        outColor += L;
    }

    if (kSamplesPerPixel > 1)
    {
        outColor *= (1.f / kSamplesPerPixel);
    }

    // TODO: Hook up debug module to allow shader asserts
    // assert(!any(isnan(outColor)));

    // Write output.
    // TODO: If !kForceAlphaOne then lookup alpha from G-buffer data.
    gOutputColor[pixel] = float4(outColor, 1.f);
}
