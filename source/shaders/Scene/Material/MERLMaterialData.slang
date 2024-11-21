/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#pragma once
#include "Utils/HostDeviceShared.slangh"

#ifdef HOST_CODE
#include "Scene/Material/TextureHandle.slang"
#include "Scene/Material/DiffuseSpecularData.slang"
#else
__exported import Scene.Material.TextureHandle;
__exported import Scene.Material.DiffuseSpecularData;
#endif

BEGIN_NAMESPACE_FALCOR

/** This is a host/device structure that describes a measured MERL material.
*/
struct MERLMaterialData
{
    // MaterialHeader (16B) is stored just before this struct in memory.
    uint bufferID = 0;                  ///< Buffer ID in material system where BRDF data is stored.
    uint samplerID = 0;                 ///< Texture sampler ID for LUT sampler.
    DiffuseSpecularData extraData = {}; ///< Parameters for a best fit BRDF approximation.
    TextureHandle texAlbedoLUT;         ///< Texture handle for albedo LUT.

    static constexpr uint kAlbedoLUTSize = 256;
};

END_NAMESPACE_FALCOR
