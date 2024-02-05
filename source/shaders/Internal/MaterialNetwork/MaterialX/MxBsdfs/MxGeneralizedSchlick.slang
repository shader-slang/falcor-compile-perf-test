/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
__exported import Rendering.Materials.PBRT.PBRTDielectricMaterialInstance;

#define DistributionGgx 0

#define ScatterModeR 0
#define ScatterModeT 1
#define ScatterModeRT 2

PBRTDielectricBSDF createMxGeneralizedSchlick(
    float3 color0,
    float3 color90,
    float exponent,
    float2 roughness,
    float3 N,
    float3 U,
    int distribution,
    int scatter_mode
)
{
    PBRTDielectricBSDF result;
    result.D.alpha = roughness;
    // result.eta = ior;
    result.eta = 1.5;

    // TODO: We want general Dielectric that does both Fresnel (and that becomes PBRTDielectric) and Schlick (used
    // here).
    return result;
}
