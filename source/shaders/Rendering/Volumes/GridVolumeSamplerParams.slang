/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

/** Enumeration of available volume transmittance estimators.
*/
enum class TransmittanceEstimator
{
    DeltaTracking,
    RatioTracking,
    RatioTrackingLocalMajorant,  // BrickedGrid only.
};

FALCOR_ENUM_INFO(TransmittanceEstimator, {
    { TransmittanceEstimator::DeltaTracking, "DeltaTracking" },
    { TransmittanceEstimator::RatioTracking, "RatioTracking" },
    { TransmittanceEstimator::RatioTrackingLocalMajorant, "RatioTrackingLocalMajorant" },
});
FALCOR_ENUM_REGISTER(TransmittanceEstimator);

enum class DistanceSampler
{
    DeltaTracking,
    DeltaTrackingLocalMajorant,  // BrickedGrid only.
};

FALCOR_ENUM_INFO(DistanceSampler, {
    { DistanceSampler::DeltaTracking, "DeltaTracking" },
    { DistanceSampler::DeltaTrackingLocalMajorant, "DeltaTrackingLocalMajorant" },
});
FALCOR_ENUM_REGISTER(DistanceSampler);

#ifdef HOST_CODE
inline bool requiresBrickedGrid(DistanceSampler d) { return d == DistanceSampler::DeltaTrackingLocalMajorant; }
inline bool requiresBrickedGrid(TransmittanceEstimator t) { return t == TransmittanceEstimator::RatioTrackingLocalMajorant; }
#endif

END_NAMESPACE_FALCOR
