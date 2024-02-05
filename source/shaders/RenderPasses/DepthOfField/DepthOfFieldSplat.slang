/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/

// Shared declarations.
// TODO: Refactor out more shared code from raster and compute passes.

struct BokehSplat
{
    uint2 color;
    uint packedScreenPos;
    uint packedRadiusLevel;

    float getRadius() { return f16tof32(packedRadiusLevel >> 16); }
    uint getLevel() { return packedRadiusLevel & 0xffff; }
};
