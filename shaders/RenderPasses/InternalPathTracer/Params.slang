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

/** Scheduling modes.
*/
enum class SchedulingMode
{
    QueueInline,        ///< Persistent threads using queues and TraceRayInline.
    SimpleInline,       ///< Simple compute pass using TraceRayInline.
    SimpleTraceRay,     ///< Simple raytracing pass using TraceRay.
    ReorderTraceRay,    ///< Raytracing pass using SER/HitObject API.
    ReorderInline,      ///< Raytracing pass using SER/HitObject API and TraceRayInline.
};

FALCOR_ENUM_INFO(SchedulingMode, {
    { SchedulingMode::QueueInline, "QueueInline" },
    { SchedulingMode::SimpleInline, "SimpleInline" },
    { SchedulingMode::SimpleTraceRay, "SimpleTraceRay" },
    { SchedulingMode::ReorderTraceRay, "ReorderTraceRay" },
    { SchedulingMode::ReorderInline, "ReorderInline" },
});
FALCOR_ENUM_REGISTER(SchedulingMode);

/** Enumeration of the atomic counters used.
*/
enum class Counters
{
    // TODO: Better naming
    kGeneralPaths = 0,
    kGeneralPathsProcessed,
    kSpecularPaths,
    kSpecularPathsProcessed,
    kDeltaReflectionPaths,
    kDeltaReflectionPathsProcessed,
    kDeltaTransmissionPaths,
    kDeltaTransmissionPathsProcessed,

    kGeneralQueued,
    kGeneralQueuedProcessed,

    kThreadCountSpecular,
    kThreadCountGeneral,
    kThreadCountDeltaReflection,
    kThreadCountDeltaTransmission,

    // Must be last
    kCount
};

/** Enumeration of warp profiling events.
*/
enum class WarpProfilerEvent
{
    MaterialCreate,
    MaterialSample,
    MaterialEval,
    MaterialEvalPdf,

    Count // Must be last
};

static const uint kMaxWarpProfilerVertices = 16; ///< Warp profiler stats for higher vertex indices are accumulated in the last bin.

/** Per-sample color format.
*/
enum class ColorFormat : uint32_t
{
    RGBA32F         = 0,
    LogLuvHDR       = 1,
};

FALCOR_ENUM_INFO(ColorFormat, {
    { ColorFormat::RGBA32F, "RGBA32F" },
    { ColorFormat::LogLuvHDR, "LogLuvHDR" },
});
FALCOR_ENUM_REGISTER(ColorFormat);

enum class MISHeuristic : uint32_t
{
    Balance     = 0,    ///< Balance heuristic.
    PowerTwo    = 1,    ///< Power heuristic (exponent = 2.0).
    PowerExp    = 2,    ///< Power heuristic (variable exponent).
};

FALCOR_ENUM_INFO(MISHeuristic, {
    { MISHeuristic::Balance, "Balance" },
    { MISHeuristic::PowerTwo, "PowerTwo" },
    { MISHeuristic::PowerExp, "PowerExp" },
});
FALCOR_ENUM_REGISTER(MISHeuristic);

// Define tile sizes in pixels.
// The frame is divided into tiles stored in scanline order, with pixels in tiles enumerated in Morton order.
static const uint2 kScreenTileDim = { 16, 16 };     ///< Screen-tile dimension in pixels.
static const uint2 kScreenTileBits = { 4, 4 };      ///< Bits needed to describe pixel position within a screen-tile.

// Define path configuration limits.
static const uint kMaxSamplesPerPixel = 16;         ///< Maximum supported sample count. We can use tiling to support large sample counts if needed.
static const uint kMaxFrameDimension = 4096;        ///< Maximum supported frame dimension in pixels along x or y. We can increase the bit allocation if needed.
static const uint kMaxBounces = 254;                ///< Maximum supported number of bounces per bounce category (value 255 is reserved for internal use). The resulting path length may be longer than this.
static const uint kMaxLightSamplesPerVertex = 8;    ///< Maximum number of shadow rays per path vertex for next-event estimation.

// Import static specialization constants.
#ifndef HOST_CODE
__exported import StaticParams;
__exported import Utils.Math.BitTricks;
#endif

/** Path tracer parameters shared between host/device code.
*/
struct PathTracerParams
{
    uint    threadCount = 82 * 64 * 16; ///< Number of persistent threads. We need to fill the machine, but it is fine to oversubscribe.
    int     useFixedSeed = false;       ///< Use fixed random seed. This is useful for debugging.
    uint    fixedSeed = 1;              ///< The seed to use when 'useFixedSeed' is enabled.
    float   lodBias = 0.f;              ///< LOD bias applied to secondary hits.

    float   specularRoughnessThreshold = 0.25f; ///< Specular reflection events are only classified as specular if the material's roughness value is equal or smaller than this threshold. Otherwise they are classified diffuse.
    float3  _pad0;

    // Runtime values
    uint2   frameDim = { 0, 0 };        ///< Frame dimension in pixels.
    uint2   screenTiles = { 0, 0 };     ///< Number of screen-tiles. Screen tiles may extend outside the frame.

    uint    frameCount = 0;             ///< Frames rendered. This is used as random seed.
    uint    seed = 0;                   ///< Random seed. This will get updated from the host depending on settings.
    float   pixelSpreadAngle = 0.0f;    ///< Used for LOD calculation.
    uint   _pad1;

#ifndef HOST_CODE
    /** Computes the offset into the tiled sample buffer for a given tile.
        The samples for all pixels are stored consecutively after this offset.
        \param[in] tile Tile coordinates.
        \return Offset into tiled sample buffer.
    */
    uint getTileOffset(const uint2 tile)
    {
        uint maxSpp = kSamplesPerPixel > 0 ? kSamplesPerPixel : kMaxSamplesPerPixel;
        uint stride = kScreenTileDim.x * kScreenTileDim.y * maxSpp;
        uint tileIdx = tile.y * screenTiles.x + tile.x;
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

        if (kSamplesPerPixel > 0)
        {
            uint tileBits = kScreenTileBits.x + kScreenTileBits.y;
            uint pixelIdx = interleave_16bit(pixel) & ((1 << tileBits) - 1); // TODO: Use interleave_8bit() if kScreenTileBits <= 4.
            return tileOffset + pixelIdx * kSamplesPerPixel;
        }
        else
        {
            return tileOffset + sampleOffset[pixel];
        }
    }
#endif
};

END_NAMESPACE_FALCOR
