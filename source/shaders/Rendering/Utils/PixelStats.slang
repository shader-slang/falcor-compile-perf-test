/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/

/** Functionality for collecting runtime stats in the path tracer.

    The host sets the following defines:

    _PIXEL_STATS_ENABLED       Nonzero if stats should be collected.

*/
__exported import PixelStatsShared;

RWTexture2D<uint> gStatsRayCount[(uint)PixelStatsRayType::Count]; // Per-pixel ray count stats.
RWTexture2D<uint> gStatsPathLength;          // Per-pixel path length.
RWTexture2D<uint> gStatsPathVertexCount;     // Per-pixel path vertex count.
RWTexture2D<uint> gStatsVolumeLookupCount;   // Per-pixel volume lookup count.

#ifdef _PIXEL_STATS_ENABLED
static uint2 gPixelStatsPixel;
#endif

void logSetPixel(uint2 pixel)
{
#ifdef _PIXEL_STATS_ENABLED
    gPixelStatsPixel = pixel;
#endif
}

void logTraceRay(PixelStatsRayType rayType)
{
#ifdef _PIXEL_STATS_ENABLED
    InterlockedAdd(gStatsRayCount[(uint)rayType][gPixelStatsPixel], 1);
#endif
}

void logPathLength(uint pathLength)
{
#ifdef _PIXEL_STATS_ENABLED
    gStatsPathLength[gPixelStatsPixel] = pathLength;
#endif
}

void logPathVertex()
{
#ifdef _PIXEL_STATS_ENABLED
    InterlockedAdd(gStatsPathVertexCount[gPixelStatsPixel], 1);
#endif
}

void logVolumeLookup()
{
#ifdef _PIXEL_STATS_ENABLED
    InterlockedAdd(gStatsVolumeLookupCount[gPixelStatsPixel], 1);
#endif
}
