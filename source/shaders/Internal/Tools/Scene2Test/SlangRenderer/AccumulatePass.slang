/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
struct AccumulateParams
{
    uint2 frameDim;
    Texture2D<float4> inTexture;
    RWTexture2D<float4> outTexture;
    uint frameIndex;
};

ParameterBlock<AccumulateParams> params;

[shader("compute")]
[numthreads(16, 16, 1)]
void accumulatePass(uint3 threadIdx: SV_DispatchThreadID)
{
    const uint2 pixel = threadIdx.xy;
    if (any(pixel >= params.frameDim))
        return;

    const float oldWeight = float(params.frameIndex - 1) / float(params.frameIndex);
    const float newWeight = 1.f / float(params.frameIndex);

    float4 oldValue = params.outTexture[pixel];
    float4 newValue = params.inTexture[pixel];

    if (params.frameIndex == 1)
        oldValue = float4(0, 0, 0, 1);

    float4 result = oldValue * oldWeight + newValue * newWeight;

    /// Just to make sure we don't get transparency via averaging
    if (oldValue[3] == 1 && newValue[3] == 1)
        result[3] = 1;

    params.outTexture[pixel] = result;
}
