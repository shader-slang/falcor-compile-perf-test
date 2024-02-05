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

#ifdef HOST_CODE
#include "Scene/Material/TextureHandle.slang"
#include "Scene/Material/MaterialData.slang"
#else
__exported import Scene.Material.TextureHandle;
__exported import Scene.Material.MaterialData;
#endif

BEGIN_NAMESPACE_FALCOR

/**
 * This is a host/device structure that describes a MDL material.
 */
struct MDLMaterialData
{
    // MaterialHeader (16B) is stored just before this struct in memory.

    /// ID of the read-only data buffer.
    uint readOnlyDataBufferID = 0;
    /// ID of the texture table buffer.
    uint textureTableBufferID = 0;
    /// ID of the argument buffer.
    uint argBlockBufferID = 0;

    /// Number of bsdf slots used by the material.
    uint surfaceScatterBsdfCount = 0;
};

struct MDLTextureInfo
{
    /// Texture handle.
    /// For 2D textures, this is a TextureHandle referencing a texture in TextureManager.
    /// For 3D textures, this is an ID referencing a 3D texture managed by MaterialSystem.
    uint texture;
};

END_NAMESPACE_FALCOR
