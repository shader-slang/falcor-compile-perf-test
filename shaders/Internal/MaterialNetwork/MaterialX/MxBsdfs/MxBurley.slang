/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
__exported import Rendering.Materials.BSDFs.DisneyDiffuseBRDF;

DisneyDiffuseBRDF createMxBurley(float3 color, float roughness, float3 normal)
{
    DisneyDiffuseBRDF result = { color, roughness };
    return result;
}
