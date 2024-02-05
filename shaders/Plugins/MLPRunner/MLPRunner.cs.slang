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

#ifndef FALCOR_ENABLE_TIN
#error Expected FALCOR_ENABLE_TIN to be defined
#endif
#ifndef FP16
#error Expected FP16 to be defined
#endif
#ifndef THREAD_GROUP_SIZE
#error Expected THREAD_GROUP_SIZE to be defined
#endif

#if FP16
typedef float16_t NetworkType;
typedef NetworkParamStorageFP16 NetworkParamStorage;
#else
typedef float NetworkType;
typedef NetworkParamStorageFP32 NetworkParamStorage;
#endif

struct MLPRunner
{
    uint batchSize;        ///< Number of network evaluations (= number of threads).
    uint paramsByteStride; ///< Stride in bytes between network parameters for each network.

    // Resources
    RWStructuredBuffer<float> result; ///< Array of outputs for each thread, indexed linearly by threadID*inputWidth.
    StructuredBuffer<float> input;    ///< Array of inputs for each thread, indexed linearly by threadID*inputWidth.
    StructuredBuffer<uint> masks;     ///< Execution mask for each warp.
    StructuredBuffer<uint> indices;   ///< Network index for each thread.

    StructuredBuffer<NetworkParamStorage> params;
    StructuredBuffer<Tin::MMAMatStore> weightsTin;
    StructuredBuffer<uint> biasTin;
    StructuredBuffer<float16_t> paramsNVCoopVec;

    void execute(const uint threadID)
    {
        if (threadID >= batchSize)
            return;

        const uint warpIdx = threadID / 32;
        const uint laneIdx = WaveGetLaneIndex();
        const uint mask = masks[warpIdx];

        NetworkType out[OUTPUT_WIDTH] = {};

        if (mask & (1u << laneIdx))
        {
#if GATHER_WEIGHTS
            // Use dynamically loaded network index. This may be uniform or non-uniform.
            const uint networkIdx = indices[threadID];
#else
            // Use uniform statically known network index.
            const uint networkIdx = 0;
#endif

            // Load inputs.
            NetworkType in[INPUT_WIDTH] = {};

            [unroll]
            for (uint i = 0; i < INPUT_WIDTH; i++)
            {
                in[i] = (NetworkType)input[threadID * INPUT_WIDTH + i];
            }

            // Evaluate network.
            TestMLP mlp;
            uint byteOffset = networkIdx * paramsByteStride;
            mlp.eval(params, weightsTin, biasTin, paramsNVCoopVec, in, out, byteOffset);
        }

        // Store outputs.
        [unroll]
        for (uint i = 0; i < OUTPUT_WIDTH; i++)
        {
            result[threadID * OUTPUT_WIDTH + i] = (float)out[i];
        }
    }
};

ParameterBlock<MLPRunner> gMLPRunner;

/**
 * The dispatch size is 64xY thread groups.
 * Each thread group is Nx1 threads where N is THREAD_GROUP_SIZE.
 */
[numthreads(THREAD_GROUP_SIZE, 1)]
void main(uint groupIndex: SV_GroupIndex, uint3 groupID: SV_GroupID)
{
    const uint threadID = (groupID.y * 64 + groupID.x) * THREAD_GROUP_SIZE + groupIndex; // Global thread index.

    gMLPRunner.execute(threadID);
}
