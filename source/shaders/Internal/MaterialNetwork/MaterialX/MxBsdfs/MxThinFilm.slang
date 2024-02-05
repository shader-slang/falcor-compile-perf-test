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

PBRTDielectricBSDF createMxThinFilm(float thickness, float ior)
{
    // TODO: Replace with proper thin film
    PBRTDielectricBSDF result;
    result.D.alpha = float(0.f);
    result.eta = ior;
    return result;
}
