/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/

/**
 * Utility functions for fp16 math.
 */

/**
 * Converts a finite fp32 number to fp16, rounding down to the nearest representable number.
 * If the result does not fit in the range of fp16, +-inf is returned, assuming the input was finite.
 * @return Rounded fp16 value in the low 16-bits. The high 16-bits are zero.
 */
uint f32tof16_roundDown(float value)
{
    uint h = f32tof16(value);
    float res = f16tof32(h); // TODO: Use precise keyword when available.
    if (res > value)
    {
        // Result was rounded up.
        // Depending on the value, the next smaller fp16 number is given by:
        // res < 0: +1 gives the next smaller
        // res > 0: -1 gives the next smaller
        // res == +-0: next smaller is 0x8001
        if (res < 0.f)
            h++;
        else if (res > 0.f)
            h--;
        else
            h = 0x8001;
    }
    return h;
}

/**
 * Converts a finite fp32 number to fp16, rounding up to the nearest representable number.
 * If the result does not fit in the range of fp16, +-inf is returned, assuming the input was finite.
 * @return Rounded fp16 value in the low 16-bits. The high 16-bits are zero.
 */
uint f32tof16_roundUp(float value)
{
    uint h = f32tof16(value);
    float res = f16tof32(h); // TODO: Use precise keyword when available
    if (res < value)
    {
        // Result was rounded down.
        // Depending on the value, the next larger fp16 number is given by:
        // res < 0: -1 gives the next larger
        // res > 0: +1 gives the next larger
        // res == +-0: next larger is 0x0001
        if (res < 0.f)
            h--;
        else if (res > 0.f)
            h++;
        else
            h = 0x0001;
    }
    return h;
}
