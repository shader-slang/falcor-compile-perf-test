/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/

// Include math constants (M_PI etc.). These are for use in this file only,
// as macro definitions are not exported from a Slang module.
#include "Utils/Math/MathConstants.slangh"

/**
 * Ray intersection against linear swept sphere using conic section formulation.
 * Steven Parker's slides: https://drive.google.com/file/d/1BkvFEwb0r4W_6VHiZvgaeJTjBlFSyp1J/view?usp=sharing
 * @param[in] rayOrigin Ray origin position.
 * @param[in] rayDir Unit ray direction vector.
 * @param[in] sphereA Sphere (3D position + radius) at one end point.
 * @param[in] sphereB Sphere at the other end point.
 * @param[in] useSphereJoints Indicating whether we test ray-sphere intersection at curve joints or not.
 * @param[out] result The closest intersection distance t, and a parameter u for linear interpolation (between 0 and 1).
 * @return True if the ray intersects the linear swept sphere segment.
 */
bool intersectLinearSweptSphereConic(
    float3 rayOrigin,
    float3 rayDir,
    float4 sphereA,
    float4 sphereB,
    bool useSphereJoints,
    out float2 result
)
{
    result = float2(FLT_MAX);
    const float3 p0 = sphereA.xyz;
    const float3 p1 = sphereB.xyz;
    const float r0 = sphereA.w;
    const float r1 = sphereB.w;

    // Perpendicular distance to closest of (p0, p1) minus the max radius.
    float t0 = min(dot(rayDir, p0 - rayOrigin), dot(rayDir, p1 - rayOrigin)) - max(r0, r1);
    t0 = max(0.f, t0);
    // For better numerical stability, push the ray to be as close as possible to the curve.
    rayOrigin += t0 * rayDir;

    // Option D: midpoint formulation.
    const float3 pMid = 0.5 * (p0 + p1);
    const float3 dirA = p1 - p0;
    const float rMid = 0.5 * (r0 + r1);
    const float rDiff = r1 - r0;
    const float3 O = rayOrigin - pMid;

    const float a = dot(rayDir, rayDir);
    const float b = dot(rayDir, dirA);
    const float c = dot(dirA, dirA) - rDiff * rDiff;
    const float d = dot(rayDir, O);
    const float e = dot(dirA, O) + rMid * rDiff;
    const float f = dot(O, O) - rMid * rMid;

    const float At = b * b - a * c;
    const float Bt = b * e - c * d;
    const float Ct = e * e - c * f;
    const float disc = Bt * Bt - At * Ct;

    // Missing the infitie cone completely.
    if (disc <= 0)
        return false;

    // Option I: t < 0 check.
    if ((Bt < 0 && Ct < 0) || (At > 0 && Bt < 0))
        return false;

    // Option K: -0.5 < u < 0.5 check.
    // Seems like there were some typos in the derivation.
    const float tmp1 = b * b * disc;
    const float tmp2 = b * Bt;
    const float tmp3 = 0.5 * c + e;
    const float tmp4 = tmp3 - c;
    const float tmp5 = At * tmp3 - tmp2;
    const float tmp6 = At * tmp4 - tmp2;

    const int sign1 = sign(c) * sign(At);
    const int sign2 = sign1 * sign(b);
    const int sign5 = sign1 * sign(tmp5);
    const int sign6 = sign1 * sign(tmp6);
    const float tmp7 = -sign2 * tmp1;
    if (tmp7 > sign6 * tmp6 * tmp6 && tmp7 < sign5 * tmp5 * tmp5)
    {
        result.x = (-Bt + sqrt(disc)) / At;
        result.y = (b * result.x + e) / c;
    }

    if (useSphereJoints)
    {
        // Option M: Knuckles and swept sphere endcaps at u = 0.5.
        const float As = a;
        const float Bs = 2.f * d - b;
        const float Cs = 0.25 * c + f - e;

        const float discSph = Bs * Bs - 4.f * As * Cs;
        if (discSph >= 0)
        {
            const float sqrtDiscSph = sqrt(discSph);
            const float inv2As = 0.5f / As;

            [unroll]
            for (uint i = 0; i < 2; i++)
            {
                float t = (-Bs + (i * 2.f - 1.f) * sqrtDiscSph) * inv2As;
                // TODO: Remove interior hits at sphere endcaps.
                if (t >= 0 && t < result.x)
                {
                    result.x = t;
                    result.y = 0.5f;
                }
            }
        }
    }

    result += float2(t0, 0.5f);
    return (result.x < FLT_MAX);
}

/**
 * Ray intersection against linear swept sphere.
 * Code is from:
 * https://gitlab-master.nvidia.com/omniverse/kit/-/blob/c2639be6382400190ef2be6a5d01a5b26e0832b9/rendering/shaders/rtx/procedural/Common.hlsl
 * OptiX code: https://p4viewer.nvidia.com/getfile///sw/wsapps/raytracing/rtsdk/playpen/hair/samples_exp/internal/hair_bench/hair_linear.cu
 * @param[in] rayOrigin Ray origin position.
 * @param[in] rayDir Unit ray direction vector.
 * @param[in] sphereA Sphere (3D position + radius) at one end point.
 * @param[in] sphereB Sphere at the other end point.
 * @param[out] result The closest intersection distance t, and a parameter u for linear interpolation (between 0 and 1).
 * @return True if the ray intersects the linear swept sphere segment.
 */
bool intersectLinearSweptSphereOptiX(float3 rayOrigin, float3 rayDir, float4 sphereA, float4 sphereB, out float2 result)
{
    result = float2(FLT_MAX);
    const float3 pa = sphereA.xyz;
    const float3 pb = sphereB.xyz;
    const float ra = sphereA.w;
    const float rb = sphereB.w;

    const float3 ab = pb - pa;
#if 1
    // early out test
    float3 dir = cross(rayDir, ab);
    dir = dir * dot(rayOrigin - pa, dir) / dot(dir, dir); // trying to use oa here made it slightly slower
    const float rmax = max(ra, rb);
    if (dot(dir, dir) > rmax * rmax)
        return false;
#endif

    float3 oa = pa - rayOrigin;
    float d_oa = dot(rayDir, oa);
    const float rab = rb - ra;

    float dpa = 0.f;
#if 1
    // precision fix -- push the ray to be as close as possible to the curve
    // TODO: optimize the dpa calculation to remove sqrtf
    // Not able to tell yet which of these is faster
    // dpa = d_oa - (length(ab) + ra + rb); // perpendicular distance to pa minus the max length of the curve from there
    dpa = min(d_oa, dot(rayDir, pb - rayOrigin)) - max(ra, rb); // perpendicular distance to closest of (pa,pb) minus the max
                                                                // radius from there

    dpa = max(0.f, dpa);
    rayOrigin += dpa * rayDir;
    oa = pa - rayOrigin;    // recalculate since ro moved
    d_oa = dot(rayDir, oa); // recalculate since ro moved
#endif

    const float E = dot(ab, ab) - rab * rab;
    const float F = dot(ab, oa) - ra * rab;
    const float G = dot(oa, oa) - ra * ra;

    const float d_ab = dot(rayDir, ab);
    const float d_ab_d_ab = d_ab * d_ab;
    const float d_ab_d_oa = d_ab * d_oa;

    const float a = E * (E - d_ab_d_ab);
    const float b = E * (d_ab_d_oa - F); // it's really -b/2, makes some constants go away
    const float c = F * (F - 2.f * d_ab_d_oa) + G * d_ab_d_ab;

    const float discrim = max(0.f, b * b - a * c); // it's necessary to clamp & continue to avoid false misses

    const float sqdiscrim = sqrt(discrim);
    const float rcp_a = 1.f / a;
    float u0 = (b - sqdiscrim) * rcp_a;
    float u1 = (b + sqdiscrim) * rcp_a;

    // If interpolation results in a negative radius, the root is out
    // of range, but we need the opposite end.  Flip it around and let
    // it get clamped to the right end.
    if (ra + u0 * rab < 0.f)
    {
        u0 = 1.f - u0;
    }
    if (ra + u1 * rab < 0.f)
    {
        u1 = 1.f - u1;
    }

    u0 = saturate(u0); // u0 = clamp(u0, 0.f, 1.f);
    u1 = saturate(u1); // u1 = clamp(u1, 0.f, 1.f);

    float t = FLT_MAX;
    float umin = u0;

    const float B0 = d_oa + u0 * d_ab;
    const float C0 = u0 * u0 * E + 2.f * u0 * F + G;
    float d0 = B0 * B0 - C0;
    if (d0 >= 0.f)
    {
        d0 = sqrt(d0);
        const float t00 = B0 - d0;
        // const float t01 = B0 + d0; // theory: unused?
        if (t00 < t && t00 > 0.f)
        {
            t = t00;
            umin = u0;
        }
        // if (t01 < t && t01 > 0.f) { t = t01; umin = u0; } // theory: unused?
    }

    // if (u1 != u0) // this test appears to hurt performance
    {
        const float B1 = d_oa + u1 * d_ab;
        const float C1 = u1 * u1 * E + 2.f * u1 * F + G;
        float d1 = B1 * B1 - C1;
        if (d1 >= 0.f)
        {
            d1 = sqrt(d1);
            const float t10 = B1 - d1;
            // const float t11 = B1 + d1; // theory: unused?
            if (t10 < t && t10 > 0.f)
            {
                t = t10;
                umin = u1;
            }
            // if (t11 < t && t11 > 0.f) { t = t11; umin = u1; } // theory: unused?
        }
    }

    result = float2(t + dpa, umin);
    return (t < FLT_MAX);
}
