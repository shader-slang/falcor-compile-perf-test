/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"

import Scene.Scene;
import Utils.Color.ColorHelpers;

cbuffer CB
{
    uint gTriangleCount;                    ///< Total number of triangles.
}

SamplerState gPointSampler;                             ///< Sampler for fetching individual texels with nearest filtering.
ByteAddressBuffer gTexelMax;                            ///< Max over texels in fp32 format.
ByteAddressBuffer gTexelSum;                            ///< Sum over texels (RGB) + number of texels (A) in 64-bit fixed-point format.
StructuredBuffer<PackedEmissiveTriangle> gTriangleData; ///< Per-triangle geometry data for emissive triangles.
RWStructuredBuffer<EmissiveFlux> gFluxData;             ///< Per-triangle flux data for emissive triangles.

uint64_t asuint64(uint lowbits, uint highbits)
{
    return (uint64_t(highbits) << 32) | uint64_t(lowbits);
}

/** Kernel computing the final pre-integrated triangle average radiance and flux.
    One dispatch with one thread per triangle (the dispatch is arranged as Y blocks of 256x1 threads).
*/
[numthreads(256, 1, 1)]
void finalizeIntegration(uint3 DTid : SV_DispatchThreadID)
{
    const uint triIdx = DTid.y * 256 + DTid.x;
    if (triIdx >= gTriangleCount) return;

    // Compute the triangle's average emitted radiance (RGB).
    // For this purpose we access the material data directly for basic materials.
    // No type checking is needed because the host side only operates on basic materials.
    // TODO: Generalize light collection to support arbitrary materials (#1314).
    const EmissiveTriangle tri = gTriangleData[triIdx].unpack();
    const BasicMaterialData materialData = gScene.materials.getBasicMaterialData(tri.materialID);
    float3 averageEmissiveColor = materialData.emissive;

    if (gScene.materials.isEmissiveTextured(tri.materialID))
    {
        // Compute the triangle's average textured emissive color based on the pre-integration results.
        // The alpha channel stores the total coverage in texels.
        // If the coverage is zero, the triangle is degenerate in texture space (line or point).
        // In that case, the emission is approximated as the average emission sampled at the three vertices.

        // Load accumulated texel values in fixed-point format.
        uint address = triIdx * 32;
        uint4 a = gTexelSum.Load4(address);
        uint4 b = gTexelSum.Load4(address + 16);
        uint64_t4 f = uint64_t4(asuint64(a.x, a.y), asuint64(a.z, a.w), asuint64(b.x, b.y), asuint64(b.z, b.w));

        // Convert from 29.35 bit fixed point.
        const float scale = (1ull << 35);
        float4 val = float4(f) / scale;

        // Rescale texel values to original range.
        float maxVal = asfloat(gTexelMax.Load(triIdx * 4));
        float3 texelSum = val.xyz * maxVal;
        float weight = val.w;

        // Compute average emissive color.
        if (weight > 0.f)
        {
            averageEmissiveColor = texelSum / weight;
        }
        else
        {
            averageEmissiveColor = float3(0.f);
            for (int i = 0; i < 3; i++)
            {
                averageEmissiveColor += gScene.materials.sampleTexture(materialData.texEmissive, gPointSampler, tri.texCoords[i], 0.f).rgb; // Sample at mip 0
            }
            averageEmissiveColor /= 3.f;
        }
    }
    float3 averageRadiance = averageEmissiveColor * materialData.emissiveFactor;

    // Pre-compute the luminous flux emitted, which is what we use during sampling to set probabilities.
    // We assume diffuse emitters and integrate per side (hemisphere) => the scale factor is pi.
    // Triangle area in m^2 (the scene units are assumed to be in meters).
    float flux = luminance(averageRadiance) * tri.area * (float) M_PI;  // Flux in lumens.

    gFluxData[triIdx].flux = flux;
    gFluxData[triIdx].averageRadiance = averageRadiance;
}
