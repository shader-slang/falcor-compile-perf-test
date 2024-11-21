/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"

/**
 * Ray type.
 * This has equal layout to DXR RayDesc but adds additional functionality.
 */
struct Ray
{
    float3 origin;
    float tMin;
    float3 dir;
    float tMax;

    /**
     * Initializes a ray.
     */
    __init(float3 origin, float3 dir, float tMin = 0.f, float tMax = FLT_MAX)
    {
        this.origin = origin;
        this.dir = dir;
        this.tMin = tMin;
        this.tMax = tMax;
    }

    /**
     * Convert to DXR RayDesc.
     */
    RayDesc toRayDesc() { return { origin, tMin, dir, tMax }; }

    /**
     * Evaluate position on the ray.
     * @param[in] t Ray parameter.
     * @return Returns evaluated position.
     */
    float3 eval(float t) { return origin + t * dir; }
};
