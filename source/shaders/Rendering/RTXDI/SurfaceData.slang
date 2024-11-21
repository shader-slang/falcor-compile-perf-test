/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import Utils.Math.PackedFormats;
import Utils.Math.FormatConversion;

__exported import PackedTypes;

/** Stores data about primary hit points.
    To reduce register pressure, some of the fields are stored in packed format
    and decoded as being accessed through Slang properties.
*/
struct SurfaceData
{
    // The following fields are stored in PackedSurfaceData.

    float3 position;        ///< Position in world-space.
    float viewDepth;        ///< Distance from viewer to surface (negative for invalid surface).
    uint packedNormal;      ///< Packed surface normal in world-space (octahedral mapping).
    uint packedWeights;     ///< Packed diffuse/specular reflectance and roughness.

    // The following fields are *NOT* stored in PackedSurfaceData.

    float3 viewDir;         ///< Direction to the viewer in world-space.

    // True if surface data is valid.
    property bool valid
    {
        get { return viewDepth >= 0.f; }
    }

    // Shading normal.
    property float3 normal
    {
        get { return decodeNormal2x16(packedNormal); }
        set { packedNormal = encodeNormal2x16(newValue); }
    }

    // Diffuse reflectance.
    property float diffuse
    {
        get { return unpackUnorm8(packedWeights); }
        set { packedWeights = (packedWeights & ~0xff) | packUnorm8(newValue); }
    }

    // Specular reflectance.
    property float specular
    {
        get { return unpackUnorm8(packedWeights >> 8); }
        set { packedWeights = (packedWeights & ~0xff00) | (packUnorm8(newValue) << 8); }
    }

    // Specular roughness.
    property float roughness
    {
        get { return unpackUnorm8(packedWeights >> 16); }
        set { packedWeights = (packedWeights & ~0xff0000) | (packUnorm8(newValue) << 16); }
    }

    // Diffuse lobe probability.
    property float diffuseProb
    {
        get { return unpackUnorm8(packedWeights >> 24); }
        set { packedWeights = (packedWeights & ~0xff000000) | (packUnorm8(newValue) << 24); }
    }

    /** Initialize an invalid surface.
    */
    __init()
    {
        this = {};
        viewDepth = -1.f;
    }

    /** Initialize a surface.
        \param[in] packed Packed surface data.
        \param[in] viewDir View direction.
    */
    __init(const PackedSurfaceData packed, const float3 viewDir)
    {
        position = asfloat(packed.position);
        viewDepth = asfloat(packed.viewDepth);
        packedNormal = packed.normal;
        packedWeights = packed.weights;
        this.viewDir = viewDir;
    }

    /** Pack surface data.
    */
    PackedSurfaceData pack()
    {
        PackedSurfaceData packed = {};
        packed.position = asuint(position);
        packed.viewDepth = asuint(viewDepth);
        packed.normal = packedNormal;
        packed.weights = packedWeights;
        return packed;
    }
};
