/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

// Define tile sizes in pixels.
// The frame is divided into tiles stored in scanline order, with pixels in tiles enumerated in Morton order.
static const uint2 kScreenTileDim = { 16, 16 }; ///< Screen-tile dimension in pixels.
static const uint2 kScreenTileBits = { 4, 4 };  ///< Bits needed to describe pixel position within a screen-tile.

// Define path configuration limits.
static const uint kMaxSamplesPerPixel = 16;  ///< Maximum supported sample count per call to execute(). TODO: check max in sample maps
static const uint kMaxFrameDimension = 4096; ///< Maximum supported frame dimension in pixels along x or y.
static const uint kThreadCount = (68 * 32 * 2) * 16; ///< 16x RTX 2080Ti. TODO: What's appropriate?

/** Enumeration of the atomic counters used.
 */
enum class Counters
{
    kVolumePaths = 0,
    kVolumePathsProcessed,
    kPathsGenerated,
    kPathsReloaded,
    kRaysScattered,
    kRaysTerminated,
    kCount // Must be last
};

#ifndef HOST_CODE
__exported import Utils.Math.BitTricks;

// Compile-time constants.
static const uint2 kResolution = uint2(RESOLUTION_X, RESOLUTION_Y);
static const uint2 kNumTiles = uint2(TILES_X, TILES_Y);
static const uint kSamplesPerPixel = SAMPLES_PER_PIXEL;
static const bool kUseAdaptiveSampling = ADAPTIVE_SAMPLING;
static const uint kMaxBounces = MAX_BOUNCES;
static const uint kPathRegenerationCount = PATH_REGENERATION_COUNT;
static const bool kShowBackground = SHOW_BACKGROUND;
static const bool kUseNEE = USE_NEE;
static const bool kUseMIS = USE_MIS;
static const bool kUseRR = USE_RR;
static const bool kComputeStats = COMPUTE_STATS;

/** Computes the offset into the tiled sample buffer for a given tile.
    The samples for all pixels are stored consecutively after this offset.
    \param[in] tile Tile coordinates.
    \return Offset into tiled sample buffer.
*/
uint getTileOffset(const uint2 tile)
{
    uint maxSpp = kUseAdaptiveSampling ? kMaxSamplesPerPixel : kSamplesPerPixel; // TODO avg spp?
    uint stride = kScreenTileDim.x * kScreenTileDim.y * maxSpp;
    uint tileIdx = tile.y * kNumTiles.x + tile.x;
    return tileIdx * stride;
}

/** Computes the offset into the tiled sample buffer for a given pixel.
    \param[in] pixel Pixel on screen.
    \param[in] sampleOffset Per-pixel sample offset within tiles. Only used if kSamplesPerPixel == 0.
    \return Offset into tiled sample buffer.
*/
uint getSampleOffset(const uint2 pixel, Texture2D<uint> sampleOffset)
{
    uint2 tileID = pixel >> kScreenTileBits;
    uint tileOffset = getTileOffset(tileID);

    if (kUseAdaptiveSampling)
    {
        return tileOffset + sampleOffset[pixel];
    }
    else
    {
        uint tileBits = kScreenTileBits.x + kScreenTileBits.y;
        uint pixelIdx = interleave_16bit(pixel) & ((1 << tileBits) - 1); // TODO: Use interleave_8bit() if kScreenTileBits <= 4.
        return tileOffset + pixelIdx * kSamplesPerPixel;
    }
}
#endif

END_NAMESPACE_FALCOR
