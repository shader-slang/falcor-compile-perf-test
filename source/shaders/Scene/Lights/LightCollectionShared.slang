/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#pragma once
#include "Utils/HostDeviceShared.slangh"

#ifdef HOST_CODE
#include "Utils/Math/PackedFormats.h"
#else
import Utils.Math.PackedFormats;
#endif

BEGIN_NAMESPACE_FALCOR

/** Per-triangle geometry data for emissive triangles.
*/
struct EmissiveTriangle
{
    float3  posW[3];            ///< Per-vertex world-space positions.
    float2  texCoords[3];       ///< Per-vertex texture coordinates.
    float3  normal;             ///< Face normal in world space.
    float   area;               ///< Triangle area in world space.
    uint    materialID;         ///< Material ID.
    uint    lightIdx;           ///< Index into global mesh lights array.

    /** Returns the interpolated position.
        \param[in] barycentrics Barycentric coordinates.
        \return Interpolated position in world space.
    */
    float3 getPosition(float3 barycentrics) CONST_FUNCTION
    {
        return posW[0] * barycentrics[0] +
               posW[1] * barycentrics[1] +
               posW[2] * barycentrics[2];
    }

    /** Returns the interpolated texture coordinate.
        \param[in] barycentrics Barycentric coordinates.
        \return Interpolated texture coordinate.
    */
    float2 getTexCoord(float3 barycentrics) CONST_FUNCTION
    {
        return texCoords[0] * barycentrics[0] +
               texCoords[1] * barycentrics[1] +
               texCoords[2] * barycentrics[2];
    }
};

/** Packed per-triangle geometry data for emissive triangles. Currently 64B.
    This struct is shared between the CPU/GPU.
*/
struct PackedEmissiveTriangle
{
    // Note: We apply the same quantization to texture coordinates for emissive geometry when loading a scene (see SceneBuilder::addMesh).
    // We should update that code whenever the quantization scheme is changed here.

    float4  posAndTexCoords[3]; ///< Per-vertex world-space positions and texture coordinates (encoded in fp16 format).
    uint    normal;             ///< Face normal in world space (encoded in 16-bit snorms, octahedral mapping).
    uint    area;               ///< Triangle area in world space (encoded in fp32 format).
    uint    materialID;         ///< Material ID.
    uint    lightIdx;           ///< Index into global mesh lights array.

    uint encodeTexCoord(float2 t) CONST_FUNCTION
    {
        uint x = f32tof16(t.x);
        uint y = f32tof16(t.y);
        return (y << 16) | x;
    }

    float2 decodeTexCoord(uint p) CONST_FUNCTION
    {
        float x = f16tof32(p & 0xffff);
        float y = f16tof32(p >> 16);
        return float2(x, y);
    }

#ifndef HOST_CODE
    [mutating] void pack(const EmissiveTriangle tri)
    {
        posAndTexCoords[0].xyz = tri.posW[0];
        posAndTexCoords[1].xyz = tri.posW[1];
        posAndTexCoords[2].xyz = tri.posW[2];
        posAndTexCoords[0].w = asfloat(encodeTexCoord(tri.texCoords[0]));
        posAndTexCoords[1].w = asfloat(encodeTexCoord(tri.texCoords[1]));
        posAndTexCoords[2].w = asfloat(encodeTexCoord(tri.texCoords[2]));
        normal = encodeNormal2x16(tri.normal);
        area = asuint(tri.area);
        materialID = tri.materialID;
        lightIdx = tri.lightIdx;
    }

    EmissiveTriangle unpack()
    {
        EmissiveTriangle tri;
        tri.posW[0] = posAndTexCoords[0].xyz;
        tri.posW[1] = posAndTexCoords[1].xyz;
        tri.posW[2] = posAndTexCoords[2].xyz;
        tri.texCoords[0] = decodeTexCoord(asuint(posAndTexCoords[0].w));
        tri.texCoords[1] = decodeTexCoord(asuint(posAndTexCoords[1].w));
        tri.texCoords[2] = decodeTexCoord(asuint(posAndTexCoords[2].w));
        tri.normal = decodeNormal2x16(normal);
        tri.area = asfloat(area);
        tri.materialID = materialID;
        tri.lightIdx = lightIdx;
        return tri;
    }
#else
    EmissiveTriangle unpack() const
    {
        EmissiveTriangle tri;
        tri.posW[0] = posAndTexCoords[0].xyz();
        tri.posW[1] = posAndTexCoords[1].xyz();
        tri.posW[2] = posAndTexCoords[2].xyz();
        tri.texCoords[0] = decodeTexCoord(asuint(posAndTexCoords[0].w));
        tri.texCoords[1] = decodeTexCoord(asuint(posAndTexCoords[1].w));
        tri.texCoords[2] = decodeTexCoord(asuint(posAndTexCoords[2].w));
        tri.normal = decodeNormal2x16(normal);
        tri.area = asfloat(area);
        tri.materialID = materialID;
        tri.lightIdx = lightIdx;
        return tri;
    }
#endif
};

/** Per-triangle flux data for emissive triangles.
    This struct is shared between the CPU/GPU.
*/
struct EmissiveFlux
{
    float   flux;               ///< Pre-integrated radiant flux emitted from the triangle.
    float3  averageRadiance;    ///< Average emitted radiance over the triangle.
};

END_NAMESPACE_FALCOR
