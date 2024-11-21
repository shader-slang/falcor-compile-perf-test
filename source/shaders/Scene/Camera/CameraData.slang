/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

/** This is a host/device structure that describes a camera.
*/
struct CameraData
{
    float4x4 viewMat;                   ///< Camera view matrix.
    float4x4 prevViewMat;               ///< Camera view matrix associated to previous frame.
    float4x4 projMat;                   ///< Camera projection matrix.
    float4x4 viewProjMat;               ///< Camera view-projection matrix.
    float4x4 invViewProj;               ///< Camera inverse view-projection matrix.
    float4x4 viewProjMatNoJitter;       ///< Camera view-projection matrix. No jittering is applied!
    float4x4 prevViewProjMatNoJitter;   ///< Camera view-projection matrix associated to previous frame. No jittering is applied!
    float4x4 projMatNoJitter;           ///< Camera projection matrix. No jittering is applied!

    float3   posW                   = float3(0, 0, 0);          ///< Camera world-space position.
    float    focalLength            = 21.0f;                    ///< Camera focal length in mm. Default is 59 degree vertical, 90 horizontal FOV at 16:9 aspect ratio.
    float3   prevPosW               = float3(0, 0, 0);          ///< Camera world-space position associated to previous frame.
    float    _padding0;
    float3   up                     = float3(0, 1, 0);          ///< Camera world-space up vector.
    float    aspectRatio            = 1.7777777f;               ///< Camera film frame aspect ratio, equal to frameWidth / frameHeight
    float3   target                 = float3(0, 0, -1);         ///< Camera target point in world-space.
    float    nearZ                  = 0.1f;                     ///< Camera near plane.
    float3   cameraU                = float3(0, 0, 1);          ///< Camera base vector U. Normalized it indicates the right image plane vector. The length is dependent on the FOV.
    float    farZ                   = 1000.0f;                  ///< Camera far plane.
    float3   cameraV                = float3(0, 1, 0);          ///< Camera base vector V. Normalized it indicates the up image plane vector. The length is dependent on the FOV.
    float    jitterX                = 0.0f;                     ///< Eventual camera jitter along the x axis expressed as a subpixel offset divided by screen width (positive value shifts the image right).
    float3   cameraW                = float3(1, 0, 0);          ///< Camera base vector W. Normalized it indicates the forward direction. The length is the camera focal distance.
    float    jitterY                = 0.0f;                     ///< Eventual camera jitter along the y axis expressed as a subpixel offset divided by screen height (positive value shifts the image up).

    float    frameHeight            = 24.0f;                    ///< Camera film frame height in mm. 24 is the height of a 35mm film
    float    frameWidth             = 42.666667f;               ///< Camera film frame width in mm.  42 2/3 is the width assuming 24mm height and a 16:9 aspect ratio
    float    focalDistance          = 10000.0f;                 ///< Camera focal distance in scene units.
    float    apertureRadius         = 0.0f;                     ///< Camera aperture radius in scene units.
    float    shutterSpeed           = 0.004f;                   ///< Camera shutter speed in seconds.
    float    ISOSpeed               = 100.0f;                   ///< Camera film speed based on ISO standards.
    float   _padding1;
    float   _padding2;
};

END_NAMESPACE_FALCOR
