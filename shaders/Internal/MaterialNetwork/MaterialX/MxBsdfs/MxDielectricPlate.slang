/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
__exported import Rendering.Materials.BSDFs.DielectricPlateBSDF;
__exported import Rendering.Materials.BSDFs.SimpleBTDF;

#define DistributionGgx 0

/// The special case for SimpleBTDF is somewhat faster, but disables optimizations for faster
#if 0
IBSDF createMxDielectricPlate(float3 tint, float ior, float2 roughness, float3 N, float3 U, int distribution)
{
    // assert(distribution == DistributionGgx);
    // assert(scatter_mode == ScatterModeR);

    if (ior == 1)
    {
        SimpleBTDF result;
        result.transmittance = float3(1.f); // could be tint, once we have it in dielectric
        return result;
    }
    else
    {
        DielectricPlateBSDF result;
        result.D.alpha = roughness;
        result.eta = 1.0 / ior; // This should probably be reverted once we have a proper MxDielectric?

        result.DInside.alpha = 2.f * ior * roughness / ((ior - 1.f) * sign(ior - 1.f));
        return result;
    }
}
#else
DielectricPlateBSDF createMxDielectricPlate(float3 tint, float ior, float2 roughness, float3 N, float3 U, int distribution)
{
    // assert(distribution == DistributionGgx);
    // assert(scatter_mode == ScatterModeR);

    DielectricPlateBSDF result;
    result.D.alpha = roughness;
    result.eta = 1.0 / ior; // This should probably be reverted once we have a proper MxDielectric?

    result.DInside.alpha = 2.f * ior * roughness / ((ior - 1.f) * sign(ior - 1.f));
    return result;
}
#endif
