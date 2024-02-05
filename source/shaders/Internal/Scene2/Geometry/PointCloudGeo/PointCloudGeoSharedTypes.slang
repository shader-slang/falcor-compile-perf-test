/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE(Falcor::scene2)

struct PointDesc
{
    float3 center;
    float radius;
};

struct PointCloudDesc
{
    uint pointOffset;
    uint pointCount;
};

#if !defined(HOST_CODE)
import Internal.Scene2.SharedTypes;

// clang-format off
extension GeometryData
{
    uint getPointCloudPointOffset() { return vbOffset; }

    uint getPointCloudPointCount() { return ibOffset; }
};
// clang-format on

#else
#include "Internal/Scene2/SharedTypes.slang"
inline void setPointCloudData(GeometryData& geometryData, uint pointOffset, uint pointCount)
{
    geometryData.vbOffset = pointOffset;
    geometryData.ibOffset = pointCount;
}
#endif

END_NAMESPACE
