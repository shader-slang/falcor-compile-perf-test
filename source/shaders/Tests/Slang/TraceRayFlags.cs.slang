/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
RWStructuredBuffer<uint> result;

[numthreads(1, 1, 1)]
void testRayFlags()
{
    result[0] = RAY_FLAG_NONE;                            // RayFlags::None
    result[1] = RAY_FLAG_FORCE_OPAQUE;                    // RayFlags::ForceOpaque
    result[2] = RAY_FLAG_FORCE_NON_OPAQUE;                // RayFlags::ForceNonOpaque
    result[3] = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH; // RayFlags::AcceptFirstHitAndEndSearch
    result[4] = RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;         // RayFlags::SkipClosestHitShader
    result[5] = RAY_FLAG_CULL_BACK_FACING_TRIANGLES;      // RayFlags::CullBackFacingTriangles
    result[6] = RAY_FLAG_CULL_FRONT_FACING_TRIANGLES;     // RayFlags::CullFrontFacingTriangles
    result[7] = RAY_FLAG_CULL_OPAQUE;                     // RayFlags::CullOpaque
    result[8] = RAY_FLAG_CULL_NON_OPAQUE;                 // RayFlags::CullNonOpaque
#ifdef DXR_1_1
    result[9] = RAY_FLAG_SKIP_TRIANGLES;              // RayFlags::SkipTriangles
    result[10] = RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES; // RayFlags::SkipProceduralPrimitives
#endif
}
