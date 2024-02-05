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

BEGIN_NAMESPACE(Falcor::scene2)

struct CurveOTSStaticVertex
{
    float3 position; // position of the vertex, first vertex of triangle is at the start of segment  --
                     // CurveTessellation::MeshResult::vertices
    float radius;    //  -- CurveTessellation::MeshResult::radii
    // I believe these two can be replaced with clever indexing tricks.
    // Namely, instead of unifying the viewDir towards start of the segment, we can point to the end of the segment.
    // At which point it could go v0-v1 is along the spine, v1-v2 is diagonal, v2-v0 is across the spine.
    // viewDir would go towards (v1+v2)/2, and the direction along the spine would be v0-v1, flipped for odd numbered primitives
    // (each slab of a cross section has two triangles, for one the v0-v1 goes along the direction of the spine,
    // and for the other it goes in the opposite direction)
    float3 fromCenter;   // direction away from the center/spine of the segment -- CurveTessellation::MeshResult::normals
    float3 curveTangent; // direction along the spine -- CurveTessellation::MeshResult::tangents
    float2 texCrd;       ///< v coordinate along the curve  -- CurveTessellation::MeshResult::texCrds
};

struct CurveOTSVertexAnimation
{
    float3 position; ///< Position is the only animated
};

enum class CurveOTSVertexType : uint16_t
{
    StrandBegin,
    StrandMid,
    StrandEnd
};

struct CurveOTSMeshDesc
{
    uint vbOffset;    ///< Offset into vertex buffer.
    uint ibOffset;    ///< Offset into index buffer, or zero if non-indexed.
    uint vertexCount; ///< Vertex count.
    uint indexCount;  ///< Index count, or zero if non-indexed.
    uint flags;       ///< Unused in the demo
    uint _pad[3];
};

#if !defined(HOST_CODE)
import Internal.Scene2.SharedTypes;

// clang-format off
extension GeometryData
{
    uint getCurveOTSVertexOffset() { return vbOffset; }

    uint getCurveOTSIndexOffset()  { return ibOffset; }
};
// clang-format on

#else
#include "Internal/Scene2/SharedTypes.slang"
inline void setCurveOTSData(GeometryData& geometryData, uint vertexOffset, uint indexOffset)
{
    geometryData.vbOffset = vertexOffset;
    geometryData.ibOffset = indexOffset;
}
#endif

END_NAMESPACE
