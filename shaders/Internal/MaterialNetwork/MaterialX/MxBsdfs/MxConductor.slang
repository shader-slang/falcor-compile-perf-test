/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
__exported import Rendering.Materials.PBRT.PBRTConductorMaterialInstance;

#define DistributionGgx 0

PBRTConductorBSDF createMxConductor(float3 ior, float3 extinction, float2 roughness, float3 normal, float3 tangent, int distribution)
{
    // assert(distribution == DistributionGgx);

    PBRTConductorBSDF result;
    result.D.alpha = roughness;
    result.eta = ior;
    result.k = extinction;
    return result;
}
