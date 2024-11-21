/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#pragma once
#include <chrono>

/**
 * Provides CPU timer utilities to the application.
 */
class CpuTimer
{
public:
    using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

    /**
     * Returns the current time.
     */
    static TimePoint getCurrentTimePoint() { return std::chrono::high_resolution_clock::now(); }

    /**
     * Update the timer.
     * @return The TimePoint of the last update() call. This value is meaningless on it's own. Call calcDuration() to get the duration that
     * passed between two TimePoints.
     */
    TimePoint update()
    {
        auto now = getCurrentTimePoint();
        mElapsedTime = now - mCurrentTime;
        mCurrentTime = now;
        return mCurrentTime;
    }

    /**
     * Get the time that passed from the last update() call to the one before that.
     * @return Elapsed time in seconds.
     */
    double delta() const { return mElapsedTime.count(); }

    /**
     * Calculate the duration in milliseconds between 2 time points.
     */
    static double calcDuration(TimePoint start, TimePoint end)
    {
        auto delta = end.time_since_epoch() - start.time_since_epoch();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(delta);
        return duration.count() * 1.0e-6;
    }

private:
    TimePoint mCurrentTime;
    std::chrono::duration<double> mElapsedTime;
};
