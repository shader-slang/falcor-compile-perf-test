/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
__exported import Internal.Rendering.Materials.BSDFs.ScratchConductorBRDF;
import Utils.Debug.PixelDebug;

#define DistributionGgx 0

ScratchConductorBRDF createMxScratchConductor(
    float3 ior,
    float3 extinction,
    float2 roughness,
    float3 scratchdirection,
    float depth,
    float3 mask,
    float scratchRoughness,
    float3 normal,
    float3 tangent,
    int distribution
)
{
    // assert(distribution == DistributionGgx);

    ScratchConductorBRDF result;
    result.D.alpha = roughness;
    result.eta = ior;
    result.k = extinction;
    result.scratchdirection = normalize(float3(scratchdirection.xy, 0.0f));
    result.depth = depth;
    result.mask = mask;
    // Convert Beckmann roughness to Phong coefficient
    result.phongCoefficient = scratchRoughness >= 0.00141421f
                                  ? -(-2.f + 2 * scratchRoughness * scratchRoughness) / (scratchRoughness * scratchRoughness)
                                  : 1000001.f;

    return result;
}
