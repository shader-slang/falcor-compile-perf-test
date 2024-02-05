/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/

[BackwardDifferentiable]
float maxFloat(float a, float b)
{
#ifndef SKIP_INVALID_GRADIENTS
    // In case we want to actually track down NaNs throughout the code, make sure that `max` will
    // actually propagate NaNs instead of hiding them. See also:
    // https://learn.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-float-rules
    if (isnan(a))
        return a;
    if (isnan(b))
        return b;
#endif
    return max(a, b);
}

[BackwardDifferentiable]
float2 maxFloat2(float2 a, float2 b)
{
    return float2(maxFloat(a.x, b.x), maxFloat(a.y, b.y));
}

[BackwardDifferentiable]
float3 maxFloat3(float3 a, float3 b)
{
    return float3(maxFloat(a.x, b.x), maxFloat(a.y, b.y), maxFloat(a.z, b.z));
}
