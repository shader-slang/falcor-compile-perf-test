/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
import Internal.Utils.Neural.TIN.TinCommon;

#if (FALCOR_ENABLE_TIN && defined(FALCOR_D3D12)) || (FALCOR_ENABLE_NV_COOP_VECTOR && defined(FALCOR_VULKAN))
import Internal.Utils.Neural.TIN.TinSIMT;
#else
import Internal.Utils.Neural.TIN.TinSIMTFallback;
#endif

#if FALCOR_ENABLE_NV_COOP_VECTOR && defined(FALCOR_VULKAN)
StructuredBuffer<float16_t> weights;
StructuredBuffer<float16_t> biases;
#else
StructuredBuffer<Tin::MMAMatStore> weights;
StructuredBuffer<uint> biases;
#endif

RWStructuredBuffer<float> result;
StructuredBuffer<uint> masks;
StructuredBuffer<float> inputs;
StructuredBuffer<uint> offsets;

cbuffer CB
{
    uint matrixByteSize;
    uint biasesByteSize;
}

#if !defined(KDIM) || !defined(NDIM)
#error KDIM and NDIM must be defined!
#endif

[numthreads(256, 1, 1)]
#if FALCOR_ENABLE_NV_COOP_VECTOR && defined(FALCOR_VULKAN)
__glsl_extension(GL_NV_cooperative_vector)
#endif
    void testVectorInit(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    const uint threadId = dispatchThreadID.x;
    const uint warpIdx = threadId / 32;
    const uint laneIdx = WaveGetLaneIndex();
    const uint mask = masks[warpIdx];

    // Initialize vector to well-defined values in all lanes.
    Tin::HCoopVector<NDIM> v = Tin::HCoopVector<NDIM>(0.h);

    for (uint i = 0; i < NDIM; i++)
    {
        v[i] = (float)(warpIdx + laneIdx + i);
    }

    // Modify vector inside dynamic control flow.
    if (mask & (1u << laneIdx))
    {
        v = v + Tin::HCoopVector<NDIM>(1.h);
    }

    for (uint i = 0; i < NDIM; i++)
    {
        result[threadId * NDIM + i] = v[i];
    }
}

[numthreads(256, 1, 1)]
void testVectorMad(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    const uint threadId = dispatchThreadID.x;
    const uint warpIdx = threadId / 32;
    const uint laneIdx = WaveGetLaneIndex();
    const uint mask = masks[warpIdx];

    // Load input from memory.
    Tin::HCoopVector<KDIM_PADDED> input = Tin::HCoopVector<KDIM_PADDED>(0.h);

    for (uint i = 0; i < KDIM; i++)
    {
        input[i] = inputs[threadId * KDIM + i];
    }

    // Initialize output to well-defined values in all lanes.
    Tin::HCoopVector<NDIM_PADDED> outputMul = Tin::HCoopVector<NDIM_PADDED>(0.h);
    Tin::HCoopVector<NDIM_PADDED> outputMad = Tin::HCoopVector<NDIM_PADDED>(0.h);

    for (uint i = 0; i < NDIM; i++)
    {
        outputMad[i] = outputMul[i] = (float)(warpIdx + laneIdx + i);
    }

    // Compute output using MUL/MAD inside dynamic control flow.
    if (mask & (1u << laneIdx))
    {
        const uint wt_offset = (offsets[threadId] * matrixByteSize) / 4;
        const uint bias_offset = (offsets[threadId] * biasesByteSize) / 4;

        outputMul = Tin::mul<KDIM_PADDED, NDIM_PADDED>(input, weights, wt_offset);
        outputMad = Tin::mad<KDIM_PADDED, NDIM_PADDED>(input, weights, wt_offset, biases, bias_offset);
    }

    for (uint i = 0; i < NDIM; i++)
    {
        result[(threadId * NDIM + i) * 2 + 0] = outputMul[i];
        result[(threadId * NDIM + i) * 2 + 1] = outputMad[i];
    }
}
