/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
import Utils.Math.HashUtils;
import GeneratedMLP; // This module is generated, not loaded from file.

#ifndef FP16
#error Expected FP16 to be defined
#endif

#if FP16
typedef float16_t NetworkType;
typedef NetworkParamStorageFP16 NetworkParamStorage;
#else
typedef float NetworkType;
typedef NetworkParamStorageFP32 NetworkParamStorage;
#endif

RWStructuredBuffer<float> result;

// Network parameters.
StructuredBuffer<NetworkParamStorage> params;
// These buffers alias the same resource as `params` above but uses different struct types.

StructuredBuffer<Tin::MMAMatStore> tinWeights;  ///< Network weights for TIN.
StructuredBuffer<uint> tinBiases;               ///< Network biases for TIN.
StructuredBuffer<float16_t> tinParamsNVCoopVec; ///< Network parameters for NV Cooperative Vector extension

StructuredBuffer<float> inputs;
StructuredBuffer<uint> indices;
StructuredBuffer<uint> masks;

cbuffer CB
{
    uint batchSize; ///< Number of network evaluations to perform (= number of threads).
}

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    const uint threadId = dispatchThreadID.x;
    if (threadId >= batchSize)
        return;

    const uint networkIdx = indices[threadId];

    NetworkType in[INPUT_WIDTH] = {};
    NetworkType out[OUTPUT_WIDTH] = {};

    // Load inputs and quantize to network precision.
    for (uint i = 0; i < INPUT_WIDTH; i++)
    {
        in[i] = (NetworkType)inputs[threadId * INPUT_WIDTH + i];
    }

    // Compute offset to network parameters in shared parameters buffer.
    const uint paramsByteOffset = networkIdx * TestMLP::kParametersByteSize;

#if USE_MASK
    if (masks[threadId])
    {
#endif
        // Evaluate network.
        // The `TestMLP` type is defined in the imported generated Slang module.
        TestMLP mlp;
        mlp.eval(params, tinWeights, tinBiases, tinParamsNVCoopVec, in, out, paramsByteOffset);

#if USE_MASK
    }
#endif

    // Store outputs converted to full precision.
    for (uint i = 0; i < OUTPUT_WIDTH; i++)
    {
        result[threadId * OUTPUT_WIDTH + i] = (float)out[i];
    }
}
