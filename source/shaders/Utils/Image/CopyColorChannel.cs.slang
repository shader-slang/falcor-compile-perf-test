/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
Texture2D<TEXTURE_FORMAT> gSrc;
RWTexture2D<TEXTURE_FORMAT> gDst;

cbuffer CB
{
    uint2 viewDim;
    uint channelIndex;
};

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    uint2 coord = dispatchThreadId.xy;
    if (any(coord >= viewDim))
        return;

    // Copy single color channel from source to all channels of destination.
    gDst[coord] = gSrc[coord][channelIndex];
}
