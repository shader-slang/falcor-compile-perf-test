/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

/** Material type.
*/
// TODO: Specify the backing type when Slang issue has been resolved
// See https://gitlab-master.nvidia.com/slang/slang/-/issues/101
enum class MaterialType // : uint32_t
{
    Unknown,    ///< Unknown material type, e.g. type is not yet registered.

    Standard,
    Cloth,
    Hair,
    MERL,
    MERLMix,
    PBRTDiffuse,
    PBRTDiffuseTransmission,
    PBRTConductor,
    PBRTDielectric,
    PBRTCoatedConductor,
    PBRTCoatedDiffuse,
    RGL,
    BuiltinCount // Must be last
};

/** Alpha mode. This specifies how alpha testing should be done.
*/
enum class AlphaMode
{
    Opaque,     ///< No alpha test.
    Mask,       ///< Alpha mask.

    Count // Must be last
};

/** Shading models supported by the standard material.
*/
enum class ShadingModel
{
    MetalRough,
    SpecGloss,

    Count // Must be last
};

/** Normal map type. This specifies the encoding of a normal map.
*/
enum class NormalMapType
{
    None,       ///< Normal map is not used.
    RGB,        ///< Normal encoded in RGB channels in [0,1].
    RG,         ///< Tangent space encoding in RG channels in [0,1].

    Count // Must be last
};

END_NAMESPACE_FALCOR
