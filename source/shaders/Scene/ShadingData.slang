/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import Scene.Material.MaterialData;
import Scene.HitInfo;
import Utils.Geometry.GeometryHelpers;
import Utils.Math.ShadingFrame;

/** This struct holds information needed for shading a hit point.

    This includes:
    - Geometric properties of the surface.
    - Texture coordinates.
    - Material ID and header.
    - Opacity value for alpha testing.
    - Index of refraction of the surrounding medium.

    Based on a ShadingData struct, the material system can be queried
    for a material instance at the hit using `gScene.materials.getMaterialInstance()`.
    The material instance has interfaces for sampling and evaluation, as well as for
    querying its properties at the hit.
*/
struct ShadingData : IDifferentiable
{
    // Geometry data
    float3  posW = float3(0.0);                 ///< Shading hit position in world space.
    float3  V = float3(0.0f);                   ///< View direction, -incident direction (-ray.dir)
    float2  uv = float2(0.0f);                  ///< Texture mapping coordinates.

    ShadingFrame frame = ShadingFrame(float3(0.0), float4(0.0));///< Smooth interpolated shading frame in world space at the shading point. The normal is *not* automatically flipped for backfacing hits.
    no_diff float3  faceN = float3(0.0);        ///< Face normal in world space, always on the front-facing side.
    no_diff float4  tangentW = float4(0.0);     ///< Geometric tangent (xyz) and sign (w) in world space. This is used for orthogonalization. Not normalized, but it is guaranteed to be nonzero and sign (w) is +-1.0.
    bool            frontFacing = false;        ///< True if primitive seen from the front-facing side, i.e., dot(V, faceN) >= 0.0.
    no_diff float   curveRadius = 0.0f;         ///< Curve cross-sectional radius. Valid only for geometry generated from curves.

    // Material data
    MaterialHeader mtl = MaterialHeader();      ///< Material header data.
    uint    materialID = 0;                     ///< Material ID at shading location.
    float   IoR = 0.0f;                         ///< Index of refraction for the medium on the front-facing side (i.e. "outside" the material at the hit).

    uint materialGradOffset = 0;                ///< Offset to the material gradient in the gradient buffer.
    uint geometryGradOffset = 0;                ///< Offset to the geometry gradient in the gradient buffer.
    uint threadID = 0;                          ///< Thread ID for gradient aggregation with a hash grid.

    // Utility functions

    /** Computes new ray origin based on the hit point to avoid self-intersection.
        The method is described in Ray Tracing Gems, Chapter 6, "A Fast and Robust
        Method for Avoiding Self-Intersection" by Carsten Wächter and Nikolaus Binder.
        \param[in] viewside True if the origin should be on the view side (reflection) or false otherwise (transmission).
        \return Ray origin of the new ray.
    */
    float3 computeRayOrigin(bool viewside = true)
    {
        return computeRayOrigin(posW, (frontFacing == viewside) ? faceN : -faceN);
    }

    /** Returns the oriented face normal.
        \return Face normal flipped to the same side as the view vector.
    */
    float3 getOrientedFaceNormal()
    {
        return frontFacing ? faceN : -faceN;
    }
};
