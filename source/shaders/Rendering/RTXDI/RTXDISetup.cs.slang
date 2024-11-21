/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/

/** This defines the various entry points for launching kernels that perform
    the main sampling/resampling operations in RTXDI:

    - Presampling local lights
    - Presampling environment light
    - Generating initial candidate samples
    - Testing visibility on initial candidate samples
    - Spatial/temporal or spatiotemporal resampling
*/

import Rendering.RTXDI.RTXDI;

cbuffer CB
{
    // Reservoir IDs for sampling/resampling entry points.
    uint gOutputReservoirID;    ///< Output reservoir.
    uint gInputReservoirID;     ///< Input reservoir (initial candidate sample or samples from previous spatial pass).
    uint gTemporalReservoirID;  ///< Temporal reservoir (samples from previous frame).
};

// Presample local lights into tiles for better memory coherency during resampling.
[numthreads(256, 1, 1)]
void presampleLocalLights(uint3 threadID : SV_DispatchThreadID)
{
    gRTXDI.presampleLocalLights(threadID.xy);
}

// Presample environment light into tiles for better memory coherency during resampling.
[numthreads(256, 1, 1)]
void presampleEnvLight(uint3 threadID : SV_DispatchThreadID)
{
    gRTXDI.presampleEnvLight(threadID.xy);
}

// Generate candidate light samples for each pixel.
[numthreads(kRTXDIGroupSize, kRTXDIGroupSize, 1)]
void generateCandidates(uint3 threadID : SV_DispatchThreadID)
{
    gRTXDI.generateCandidates(threadID.xy, gOutputReservoirID);
}

// Test visibility for the single selected light candidate for each pixel.
[numthreads(kRTXDIGroupSize, kRTXDIGroupSize, 1)]
void testCandidateVisibility(uint3 threadID : SV_DispatchThreadID)
{
    gRTXDI.testCandidateVisibility(threadID.xy, gOutputReservoirID);
}

// Perform spatial only resampling across a frame.
[numthreads(kRTXDIGroupSize, kRTXDIGroupSize, 1)]
void spatialResampling(uint3 threadID : SV_DispatchThreadID)
{
    gRTXDI.doSpatialResampling(threadID.xy, gInputReservoirID, gOutputReservoirID);
}

// Perform temporal only resampling (looking at last frame's backprojected sample).
[numthreads(kRTXDIGroupSize, kRTXDIGroupSize, 1)]
void temporalResampling(uint3 threadID : SV_DispatchThreadID, uint3 groupID : SV_GroupThreadID)
{
    gRTXDI.doTemporalResampling(threadID.xy, groupID.xy, gInputReservoirID, gTemporalReservoirID, gOutputReservoirID);
}

// Perform spatiotemporal resampling (looking at last frame's backprojected sample and a neighborhood around it).
[numthreads(kRTXDIGroupSize, kRTXDIGroupSize, 1)]
void spatiotemporalResampling(uint3 threadID : SV_DispatchThreadID, uint3 groupID : SV_GroupThreadID)
{
    gRTXDI.doSpatiotemporalResampling(threadID.xy, groupID.xy, gInputReservoirID, gTemporalReservoirID, gOutputReservoirID);
}
