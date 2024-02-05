/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
RWStructuredBuffer<uint> result;

StructuredBuffer<uint> indices;
StructuredBuffer<uint> enabled;

uint warpIteration(const uint threadIdx, const uint index)
{
    uint res = 0;

    if (WaveActiveCountBits(true) == WaveGetLaneCount())
    {
        // We have a full warp (all lanes active).
        // Proceeed by iteraing over the unique indices across the warp.
        // The loop is written so that in each iteration we have a full warp operating on a uniform index.
        uint mask = WaveMatch(index).x;
        mask &= -mask;                // Keep least significant bit in each mask.
        mask = WaveActiveBitOr(mask); // Coalesce masks across warp.

        // Setup input. The input is non-uniform across the warp.
        uint in = 3 * threadIdx + 1;

        // Iterate over set bits in mask from low to high.
        // In each iteration the lowest bit is cleared.
        for (; mask; mask &= (mask - 1))
        {
            uint laneIdx = firstbitlow(mask);
            uint idx = WaveReadLaneAt(index, laneIdx); // Broadcast index across warp.

            // Evaluate function with full warp.
            // We include `WaveActiveCountBits` to test that the warp is full.
            uint out = in * idx * WaveActiveCountBits(true);

            // Accumulate output for active lanes.
            if (idx == index)
            {
                res += out;
            }
        }
    }
    else
    {
        // We have a partial warp (some lanes inactive).
        // Set a debug value for all active lanes for verification.
        res = 0xdeaddead;
    }

    return res;
}

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    const uint i = dispatchThreadID.x;
    const uint index = indices[i];

    if (enabled[i])
    {
        // The index is non-uniform across the warp. One or more lanes may have the same index.
        // Some warps will have all lanes active here, some will be partial.
        result[i] = warpIteration(i, index);
    }
    else
    {
        // Thread is marked as disabled. Write a debug value for verification.
        result[i] = 0xd0d0d0d0;
    }
}
