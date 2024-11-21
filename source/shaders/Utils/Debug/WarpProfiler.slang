/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/

/// Import this module to use the warp profiler.
/// The shader program must be compiled for shader model 6.5 or above.
/// The implementation currently assumes 32-wide warps for NVIDIA GPUs.

/// Global buffer for collecting warp histograms.
RWStructuredBuffer<uint> gWarpHistogram;

/**
 * @brief Call to measure the utilization in the current warp.
 * The utilization is measured as the number of active threads in the warp.
 * The histogram of active thread count can later be queried on the host.
 * @param[in] binIndex Which profiling bin to use.
 */
void warpUtilization(uint binIndex)
{
    uint activeCount = WaveActiveCountBits(true);
    if (WaveIsFirstLane())
    {
        binIndex = binIndex * 32 + activeCount - 1;
        InterlockedAdd(gWarpHistogram[binIndex], 1);
    }
}

/**
 * @brief Call to measure the amount of divergence in the current warp.
 * The divergence is measured as the number of different values seen among the active threads in the warp.
 * The histogram of divergence can later be queried on the host.
 * @param[in] binIndex Which profiling bin to use.
 * @param[in] value Value for which divergence is measured.
 */
void warpDivergence(uint binIndex, uint value)
{
    // We use wave match to identify each group of distinct values across the warp.
    // The first lane in each such group is active in the branch below and `warpUtilization()`
    // is called to count the number of active threads and record the result.
    uint groupMask = WaveMatch(value).x;
    uint firstLaneInGroup = firstbitlow(groupMask);
    if (WaveGetLaneIndex() == firstLaneInGroup)
    {
        warpUtilization(binIndex);
    }
}
