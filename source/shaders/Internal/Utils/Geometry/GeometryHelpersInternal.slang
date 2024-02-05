/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
import Scene.SceneTypes;

float computeNormalCurvature(
    const float3 position[3],
    const float3 normal[3],
    const float3 barycentrics,
    const bool isFrontFaceCW,
    const float4x4 worldMat,
    const float3x3 worldInvTransposeMat,
    const float3 v
)
{
    float3 x0 = mul(worldMat, float4(position[0], 1.0f)).xyz;
    float3 x1 = mul(worldMat, float4(position[1], 1.0f)).xyz;
    float3 x2 = mul(worldMat, float4(position[2], 1.0f)).xyz;

    float3 n0 = normalize(mul(worldInvTransposeMat, normal[0]));
    float3 n1 = normalize(mul(worldInvTransposeMat, normal[1]));
    float3 n2 = normalize(mul(worldInvTransposeMat, normal[2]));

    float3 n = {};
    n += n0 * barycentrics[0];
    n += n1 * barycentrics[1];
    n += n2 * barycentrics[2];

    float normalScale = rcp(length(n));
    n = normalize(n);

    float3 A1 = x1 - x0;
    float3 A2 = x2 - x0;

    float3 Nt = cross(A1, A2);
    Nt = isFrontFaceCW ? -Nt : Nt;
    float scale = 1.f / dot(Nt, Nt);
    float3 Q1_t = cross(A2, Nt) * scale;
    float3 Q2_t = cross(Nt, A1) * scale;

    if (dot(n, v) < 0.f)
    {
        n = -n;
        n0 = -n0;
        n1 = -n1;
        n2 = -n2;
    }

    float3 nu = n1 - n0;
    float3 nv = n2 - n0;
    float3 t = v - n * dot(n, v);

    scale = normalScale / max(dot(t, t), 1e-6f);

    float normalCurvature = (dot(t, nu) * dot(Q1_t, t) + dot(t, nv) * dot(Q2_t, t)) * scale;

    // TODO: Sometimes we get NaNs but the real root cause is unknown. Maybe dot(n, v) ~= 0.0? Investigate this at some point.
    normalCurvature = (isnan(normalCurvature) || isinf(normalCurvature)) ? 0.f : normalCurvature;

    return normalCurvature;
}

/**
 * Compute normal curvature.
 * @param[in] vertices Unpacked vertices.
 * @param[in] barycentrics Barycentric coordinates in the triangle.
 * @param[in] isFrontFaceCW True if front-facing side has clockwise winding in object space.
 * @param[in] worldMat World transformation matrix.
 * @param[in] worldInvTransposeMat Inverse transpose of world transformation matrix.
 * @param[in] v Normalized view direction.
 * @return Normal curvature (positive if the circle of curvature is on the opposite side than normal).
 */
float computeNormalCurvature(
    const StaticVertexData vertices[3],
    const float3 barycentrics,
    const bool isFrontFaceCW,
    const float4x4 worldMat,
    const float3x3 worldInvTransposeMat,
    const float3 v
)
{
    const float3 positions[3] = { vertices[0].position, vertices[1].position, vertices[2].position };
    const float3 normals[3] = { vertices[0].normal, vertices[1].normal, vertices[2].normal };
    return computeNormalCurvature(positions, normals, barycentrics, isFrontFaceCW, worldMat, worldInvTransposeMat, v);
}
