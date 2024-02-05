/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
#include "RenderPasses/LegacyPathTracer/PathTracer/RayFootprintModes.slangh"
__exported import Rendering.Materials.TexLODHelpers;
__exported import Rendering.Materials.TexLODTypes;

// Check for required defines that must set to use this module.
#ifndef RAY_FOOTPRINT_MODE
#error "RAY_FOOTPRINT_MODE not defined!"
#endif

/** Ray footprint base structure for tex LOD.
    TODO: Abstract and use interface?
*/
struct RayFootprint
{
#if RAY_FOOTPRINT_MODE == RAY_FOOTPRINT_MODE_RAYCONE // Ray cone.
    RayCone rayCone;
#elif (RAY_FOOTPRINT_MODE == RAY_FOOTPRINT_MODE_RAYDIFFS_ISO) || (RAY_FOOTPRINT_MODE == RAY_FOOTPRINT_MODE_RAYDIFFS_ANISO) // Ray Diff.
    RayDiff rayDiff;
#else
    const uint dummyFootprint = 0; // Necessary, otherwise SLANG compilation crashes with empty struct.
#endif
};
