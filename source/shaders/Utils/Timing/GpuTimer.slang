/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#include "Utils/NVAPI.slangh" // We need this for accessing the global timer.

/**
 * GPU timer based on NVAPI.
 * Uses only the lower 32-bits of the global timer. This should be sufficient to do
 * most measurements but care needs to be taken for handling wrap around.
 */
struct GpuTimer
{
    typedef uint TimePoint;

    TimePoint startTime; ///< Time when timer was last started.

    /**
     * Starts the timer.
     */
    [mutating]
    void start() { startTime = getCurrentTimePoint(); }

    /**
     * Returns the elapsed time since the last call to start().
     * @return Returns the elapsed time.
     */
    uint getElapsed()
    {
        TimePoint currentTime = getCurrentTimePoint();
        return calcDuration(startTime, currentTime);
    }

    /**
     * Get the current time point.
     */
    static TimePoint getCurrentTimePoint()
    {
#if FALCOR_NVAPI_AVAILABLE
        return (TimePoint)NvGetSpecial(NV_SPECIALOP_GLOBAL_TIMER_LO);
#else
        return 0;
#endif
    }

    /**
     * Compute the duration between two time points accounting for at most one overflow.
     * @param[in] start Start time point
     * @param[in] end End time point
     * @return Returns the duration.
     */
    static uint calcDuration(TimePoint start, TimePoint end) { return end - start; }
};
