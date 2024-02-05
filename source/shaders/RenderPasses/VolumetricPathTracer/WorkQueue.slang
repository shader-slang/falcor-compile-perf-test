/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
import Params;

/** Queueing resources and helper functions.
 */
struct WorkQueue
{
    RWByteAddressBuffer counters; ///< Atomic counters used for queueing operations.
    // TODO: This only has to be RW in the generate paths pass. Make read-only for later passes.
    RWByteAddressBuffer volumePathIDs;       ///< List of volume path IDs.
    RWByteAddressBuffer volumePathDistances; ///< List of volume path distances.

    /** Called from within control flow to increment a counter atomically.
        This is useful for appending items to a list.
        \param[out] newValue The new counter value. This is uniform across the warp.
        \return The unique value assigned to each active thread.
    */
    uint incrementCounter(uint counterID, out uint newValue)
    {
        uint laneCount = WaveActiveCountBits(true);
        uint laneOffset = WavePrefixCountBits(true);
        uint originalValue;
        if (WaveIsFirstLane())
        {
            counters.InterlockedAdd(counterID * 4, laneCount, originalValue);
        }
        originalValue = WaveReadLaneFirst(originalValue); // Broadcast to all active threads
        newValue = originalValue + laneCount;
        return originalValue + laneOffset;
    }

    /** Called from within control flow to increment a counter atomically.
        This is useful for appending items to a list.
        \return The unique value assigned to each active thread.
    */
    uint incrementCounter(uint counterID)
    {
        uint newValue;
        return incrementCounter(counterID, newValue);
    }

    /** Called from within control flow to add to a counter atomically.
        This is useful for appending multiple items per thread to a list.
        \return The unique counter value assigned to each active thread.
    */
    uint addCounter(uint counterID, uint value)
    {
        uint sum = WaveActiveSum(value);
        uint offset = WavePrefixSum(value);
        uint originalValue;
        if (WaveIsFirstLane())
        {
            counters.InterlockedAdd(counterID * 4, sum, originalValue);
        }
        originalValue = WaveReadLaneFirst(originalValue); // Broadcast to all active threads
        return originalValue + offset;
    }
}
