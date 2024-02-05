/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/

/** Enums for adaptive sampler -- see AdaptiveSampler.cs.slang and AdaptiveSampler.cpp|h
 */

#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

/** This enum is shared between CPU/GPU.
    It enumerates the different modes for adaptive sampling.
*/
enum class AdaptiveSamplerMode
{
    Thresholding = 0,
    BlueNoiseDithered2D = 1,
    BlueNoiseDithered2Dx1D = 2,
};

FALCOR_ENUM_INFO(
    AdaptiveSamplerMode,
    {
        { AdaptiveSamplerMode::Thresholding, "Thresholding" },
        { AdaptiveSamplerMode::BlueNoiseDithered2D, "BlueNoiseDithered2D" },
        { AdaptiveSamplerMode::BlueNoiseDithered2Dx1D, "BlueNoiseDithered2Dx1D" },
    }
);
FALCOR_ENUM_REGISTER(AdaptiveSamplerMode);

enum class AdaptiveSamplerAnimMode
{
    Still = 0,
    Animated = 1,
    GoldenRatioAnimated = 2,
};

FALCOR_ENUM_INFO(
    AdaptiveSamplerAnimMode,
    {
        { AdaptiveSamplerAnimMode::Still, "Still" },
        { AdaptiveSamplerAnimMode::Animated, "Animated" },
        { AdaptiveSamplerAnimMode::GoldenRatioAnimated, "GoldenRatioAnimated" },
    }
);
FALCOR_ENUM_REGISTER(AdaptiveSamplerAnimMode);

enum class AdaptiveSamplerClampMode
{
    Standard = 0,       // The least expensive method in terms of compute & bandwidth, but could result in too low average SPP.
    Addition = 1,       // Adds a constant, which is computed, to each non-clamped pixel.
    Multiplication = 2, // Updates the averageSPP for each non-clamped pixel. Best method.
};

FALCOR_ENUM_INFO(
    AdaptiveSamplerClampMode,
    {
        { AdaptiveSamplerClampMode::Standard, "Standard" },
        { AdaptiveSamplerClampMode::Addition, "Addition" },
        { AdaptiveSamplerClampMode::Multiplication, "Multiplication" },
    }
);
FALCOR_ENUM_REGISTER(AdaptiveSamplerClampMode);

END_NAMESPACE_FALCOR
