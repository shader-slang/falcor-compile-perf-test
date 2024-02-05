/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
import Plugins.MaterialDataGenerator.MaterialDataGenerator;

static const uint32_t kThreadGroupSize = THREAD_GROUP_SIZE;

ParameterBlock<MaterialDataGenerator> gMaterialDataGenerator;

RWStructuredBuffer<MaterialSample> sampleBuffer;

[numthreads(kThreadGroupSize, kThreadGroupSize, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    uint4 gridDim = gMaterialDataGenerator.gridDim;
    uint sampleCount = gMaterialDataGenerator.sampleCount;
    if (any(dispatchThreadId >= uint3(gridDim.x * gridDim.z, gridDim.y * gridDim.w, sampleCount)))
        return;

    // Map thread ID to texel grid and directional grid cell indices.
    // The samples are currently written in scanline order over directional grid cells,
    // followed by scanline order over texels. We may want to change this to a Morton order.
    uint2 uvIdx = dispatchThreadId.xy / gridDim.zw;
    uint2 wiIdx = dispatchThreadId.xy % gridDim.zw;
    uint msIdx = uvIdx.y * gridDim.x + uvIdx.x;
    msIdx *= gridDim.z * gridDim.w;
    msIdx += wiIdx.y * gridDim.z + wiIdx.x;

    float2 uvFixed = sampleBuffer[msIdx].uv;
    float3 wiFixed = sampleBuffer[msIdx].wi;
    float3 woFixed = sampleBuffer[msIdx].wo;

    MaterialSample ms = gMaterialDataGenerator.generateMaterialSample(
        dispatchThreadId,
        uint2(kThreadGroupSize),
        SampleBufferContentDesc(gMaterialDataGenerator.sampleBufferContent),
        uvFixed,
        wiFixed,
        woFixed
    );
    sampleBuffer[msIdx] = ms;
}
