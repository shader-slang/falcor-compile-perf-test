/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/

/** Enumeration of available debug outputs.
    Note: Keep in sync with definition in ScreenSpaceReSTIR.h
*/
enum class DebugOutput
{
    Disabled,
    Position,
    Depth,
    Normal,
    FaceNormal,
    DiffuseWeight,
    SpecularWeight,
    SpecularRoughness,
    PackedNormal,
    PackedDepth,
    InitialWeight,
    TemporalReuse,
    SpatialReuse,
    FinalSampleDir,
    FinalSampleDistance,
    FinalSampleLi,
};
