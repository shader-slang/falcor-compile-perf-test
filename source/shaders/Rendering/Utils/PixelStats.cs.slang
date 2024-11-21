/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import PixelStatsShared;

cbuffer CB
{
    uint2 gFrameDim;
}

Texture2D<uint> gStatsRayCount[(uint)PixelStatsRayType::Count]; // Per-pixel ray count stats.
RWTexture2D<uint> gStatsRayCountTotal;

/** Compute pass for computing per-pixel total ray count.
*/
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    if (any(pixel > gFrameDim)) return;

    uint totalRays = 0;
    for (uint i = 0; i < (uint)PixelStatsRayType::Count; i++)
    {
        totalRays += gStatsRayCount[i][pixel];
    }
    gStatsRayCountTotal[pixel] = totalRays;
}
