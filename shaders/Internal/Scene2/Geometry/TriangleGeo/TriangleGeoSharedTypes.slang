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

struct TriangleStaticVertex
{
    float3 position;
    float3 normal;
    float2 texCrd;
    float3 tangent;
};

struct TriangleGeoDesc
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
    uint getTriangleVertexOffset() { return vbOffset; }

    uint getTriangleIndexOffset() { return ibOffset; }
};
// clang-format on

#else
#include "Internal/Scene2/SharedTypes.slang"
inline void setTriangleData(GeometryData& geometryData, uint vertexOffset, uint indexOffset)
{
    geometryData.vbOffset = vertexOffset;
    geometryData.ibOffset = indexOffset;
}
#endif

END_NAMESPACE
