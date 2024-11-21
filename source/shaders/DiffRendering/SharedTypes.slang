/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

enum class DiffMode : uint32_t
{
    Primal,            // Outputs primal images like path tracers do.
    BackwardDiff,      // Outputs gradient vectors.
    ForwardDiffDebug,  // Outputs gradient images w.r.t. a single variable for debugging.
    BackwardDiffDebug, // Outputs gradient images w.r.t. a single variable for debugging.
};

FALCOR_ENUM_INFO(
    DiffMode,
    {
        { DiffMode::Primal, "Primal" },
        { DiffMode::BackwardDiff, "BackwardDiff" },
        { DiffMode::ForwardDiffDebug, "ForwardDiffDebug" },
        { DiffMode::BackwardDiffDebug, "BackwardDiffDebug" },
    }
);
FALCOR_ENUM_REGISTER(DiffMode);

// We will probably add more gradient types in the future.
enum class GradientType : uint32_t
{
    Material,
    MeshPosition,
    MeshNormal,
    MeshTangent,
    Count,
};

FALCOR_ENUM_INFO(
    GradientType,
    {
        { GradientType::Material, "Material" },
        { GradientType::MeshPosition, "MeshPosition" },
        { GradientType::MeshNormal, "MeshNormal" },
        { GradientType::MeshTangent, "MeshTangent" },
    }
);
FALCOR_ENUM_REGISTER(GradientType);

enum class GradientAggregateMode : uint32_t
{
    Direct,
    HashGrid,
};

// For debugging differentiable path tracers by visualizing gradient images.

enum class DiffVariableType : uint32_t
{
    None,
    Material,
    GeometryTranslation,
};

FALCOR_ENUM_INFO(
    DiffVariableType,
    {
        { DiffVariableType::None, "None" },
        { DiffVariableType::Material, "Material" },
        { DiffVariableType::GeometryTranslation, "GeometryTranslation" },
    }
);
FALCOR_ENUM_REGISTER(DiffVariableType);

struct DiffDebugParams
{
    DiffVariableType varType;
    uint2 id;
    uint32_t offset;

    float4 grad;
};

// For inverse rendering optimization.

struct InverseOptimizationParams
{
    uint32_t meshID;
    uint3 _pad0;
};

END_NAMESPACE_FALCOR
