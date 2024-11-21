/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

/** Struct representing a material texture handle on the GPU.

    A texture handle can be in different modes:
    - 'Uniform' handle refers to a constant value.
    - 'Texture' handle refers to a traditional texture.

    In the future we'll add a 'Procedural' mode here, where the handle
    refers to a procedural texture identified by a unique ID.
*/
struct TextureHandle
{
    uint32_t packedData = 0;

    enum class Mode
    {
        Uniform,
        Texture,

        Count // Must be last
    };

    static constexpr uint kTextureIDBits = 29;
    static constexpr uint kModeBits = 2;
    static constexpr uint kUdimEnabledBits = 1;

    static constexpr uint kModeOffset = kTextureIDBits;
    static constexpr uint kUdimEnabledOffset = kModeOffset + kModeBits;

#ifdef HOST_CODE
    TextureHandle() = default;
    explicit TextureHandle(uint32_t packedData_) : packedData(packedData_) {}
#else
    __init(uint32_t packedData) { this.packedData = packedData; }
#endif

    /** Set mode.
    */
    SETTER_DECL void setMode(Mode mode) { packedData = PACK_BITS(kModeBits, kModeOffset, packedData, (uint)mode); }

    /** Get mode.
    */
    Mode getMode() CONST_FUNCTION { return Mode(EXTRACT_BITS(kModeBits, kModeOffset, packedData)); }

    /** Set texture ID. This sets mode to Mode::Texture.
     */
    SETTER_DECL void setTextureID(uint texID) { setMode(Mode::Texture); packedData = PACK_BITS(kTextureIDBits, 0, packedData, texID); }

    /** Get texture ID. This operation is only valid if mode is Mode::Texture.
    */
    uint getTextureID() CONST_FUNCTION { return EXTRACT_BITS(kTextureIDBits, 0, packedData); }

    /** Set whether the texture uses udim or not.
     */
    SETTER_DECL void setUdimEnabled(bool udimEnabled) { packedData = PACK_BITS(kUdimEnabledBits, kUdimEnabledOffset, packedData, udimEnabled ? 1 : 0); }

    /** Set whether the texture uses udim or not.
     */
    bool getUdimEnabled() CONST_FUNCTION { return IS_BIT_SET(packedData, kUdimEnabledOffset); }

#ifdef HOST_CODE
    bool operator==(const TextureHandle& rhs) const { return packedData == rhs.packedData; }
    bool operator!=(const TextureHandle& rhs) const { return packedData != rhs.packedData; }
#endif
};

/** Struct storing information about a material texture.
*/
struct TextureInfo
{
    uint width;
    uint height;
    uint depth;
    uint mipLevels;
};

END_NAMESPACE_FALCOR
