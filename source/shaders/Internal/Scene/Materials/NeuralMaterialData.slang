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
#include "NeuralMaterialTypes.slangh"

#ifdef HOST_CODE
#include "Scene/Material/TextureHandle.slang"
#include "Scene/Material/MaterialData.slang"
#else
__exported import Scene.Material.TextureHandle;
__exported import Scene.Material.MaterialData;
#endif

BEGIN_NAMESPACE_FALCOR

/**
 * This is a host/device structure that describes a neural material.
 */
struct NeuralMaterialData
{
    // MaterialHeader (16B) is stored just before this struct in memory.
    uint flags = 0;                               ///< Material flags and packed sampler IDs. See accessors below.
    uint paramsByteOffsetEvaluation = 0;          ///< Offset in bytes to parameters for evaluation network.
    uint paramsByteOffsetShFrame = 0;             ///< Offset in bytes to parameters for shading frame network.
    uint paramsByteOffsetImportanceSampling1 = 0; ///< Offset in bytes to parameters for importance sampling network 1.
    uint paramsByteOffsetImportanceSampling2 = 0; ///< Offset in bytes to parameters for importance sampling network 2.

    // Texture handles (4B each).
    TextureHandle texNeural[2];

    static constexpr uint kLatentSamplerIDOffset = 0;

    SETTER_DECL void setLatentSamplerID(uint samplerID)
    {
        flags = PACK_BITS(MaterialHeader::kSamplerIDBits, kLatentSamplerIDOffset, flags, samplerID);
    }
    uint getLatentSamplerID() CONST_FUNCTION { return EXTRACT_BITS(MaterialHeader::kSamplerIDBits, kLatentSamplerIDOffset, flags); }
};

END_NAMESPACE_FALCOR
