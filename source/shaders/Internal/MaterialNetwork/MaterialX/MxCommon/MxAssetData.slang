/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
#pragma once
#include "Utils/HostDeviceShared.slangh"
BEGIN_NAMESPACE_FALCOR

/** This is a struct that contains general per-asset data that needs to be available
    to every MaterialX shader.
    For example, transforms that place the object into BLAS go here,
    as MaterialX assumes it can get WorldFromObject matrix, and we currently
    only get WorldFromBLAS. If the object was transformed before being added into BLAS,
    we need to multiply the instanceFromObject as well. (BLAS and Instance being the same thing)
 */
struct MxAssetData
{
    // transforms, in case object gets transformed into BLAS
    float4x4 instanceFromObject = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
    float4x4 objectFromInstance = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
};

END_NAMESPACE_FALCOR
