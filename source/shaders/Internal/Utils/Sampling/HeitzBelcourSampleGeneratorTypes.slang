/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/

/**
 * Enums for HeitzBelcourSampleGenerator -- see HeitzBelcourSort.cs.slang and HeitzBelcourSampleGenerator.cpp|h
 */

#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

/**
 * This enum is shared between CPU/GPU.
 * It enumerates the different modes for adaptive sampling.
 */
enum class NoiseTarget
// TODO: Remove the ifdefs and the include when Slang supports enum type specifiers.
#ifdef HOST_CODE
    : uint32_t
#endif
{
    BlueNoise2D = 0,
    BlueNoise2Dx1D = 1,
    IGN = 2,
};

END_NAMESPACE_FALCOR
