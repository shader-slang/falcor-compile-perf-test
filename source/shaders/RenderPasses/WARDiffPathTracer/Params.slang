/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

// Import static specialization constants.
#ifndef HOST_CODE
__exported import RenderPasses.WARDiffPathTracer.StaticParams;
__exported import Utils.Math.BitTricks;
#endif

// Define path configuration limits.

/// Maximum supported sample count. We can use tiling to support large sample counts if needed.
static const uint kMaxSamplesPerPixel = 16;
/// Maximum supported frame dimension in pixels along x or y. We can increase the bit allocation if needed.
static const uint kMaxFrameDimension = 4096;
/// Maximum supported number of bounces per bounce category (value 255 is reserved for internal use).
/// The resulting path length may be longer than this.
static const uint kBounceLimit = 254;
/// Maximum number of shadow rays per path vertex for next-event estimation.
static const uint kMaxLightSamplesPerVertex = 8;

/**
 * Path tracer parameters shared between host/device code.
 */
struct WARDiffPathTracerParams
{
    /// Use fixed random seed. This is useful for debugging.
    int useFixedSeed = false;
    /// The seed to use when 'useFixedSeed' is enabled.
    uint fixedSeed = 1;
    /// Threshold for asserting on NaNs.
    float assertThreshold = 1e9f;
    /// A runtime varialbe to control running the backward pass or the primal pass.
    uint runBackward = 1;

    // Runtime values

    /// Frame dimension in pixels.
    uint2 frameDim = { 0, 0 };
    /// Number of screen-tiles. Screen tiles may extend outside the frame.
    uint2 screenTiles = { 0, 0 };

    /// Frames rendered. This is used as random seed.
    uint frameCount = 0;
    /// Random seed. This will get updated from the host depending on settings.
    uint seed = 0;
    uint2 _pad0;
};

END_NAMESPACE_FALCOR
