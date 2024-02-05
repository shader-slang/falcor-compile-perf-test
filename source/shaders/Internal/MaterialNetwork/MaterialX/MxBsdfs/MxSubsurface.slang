/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
__exported import Rendering.Materials.BSDFs.OrenNayarBRDF;

OrenNayarBRDF createMxSubsurface(float3 color, float3 radius, float anisotropy, float3 normal)
{
    // TODO: Needs a master plan for proper textured subsurface
    OrenNayarBRDF result = { color, 1.f };
    return result;
}
