/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"

import Scene.SDFs.SDFVoxelCommon;

struct SDFVoxelHitUtils
{
    /** Intersects a ray against the implicit surface surface defined by trilinearly interpolating signed distance values stored at the corner of a voxel.
        \param[in] rayOrigLocal The origin of the ray, in voxel local space, should be inside the voxel.
        \param[in] rayDirLocal The direction of the ray, in voxel local space.
        \param[in] rayOrigInsideVoxel True if the world ray origin is inside the voxel, note that this is note the same as rayOrigLocal.
        \param[in] values0xx The SDF corner values at x=0.
        \param[in] values1xx The SDF corner values at x=1.
        \param[in] tMax The max value of t, in voxel local space.
        \param[in] solverMaxStepCount If a numeric intersection algorithm is seleceted, this is the max number of steps that the algorithm is allowed to take.
        \param[out] t The distance from ray origin to the intersection point in voxel local space, if there is one, otherwise this is undefined.
        \return true if the ray hits the implicit surface, false otherwise.
    */
    static bool intersectSDFVoxel(
        const float3 rayOrigLocal,
        const float3 rayDirLocal,
        const bool rayOrigInsideVoxel,
        const float4 values0xx,
        const float4 values1xx,
        const float tMax,
        const uint solverMaxStepCount,
        out float t)
    {
#if SCENE_SDF_VOXEL_INTERSECTION_METHOD == SCENE_SDF_VOXEL_SPHERE_TRACING
        const float minstep = 0.0001f;
        return Solvers::sphereTraceVoxel(rayOrigLocal, rayDirLocal, values0xx, values1xx, minstep, tMax, solverMaxStepCount, t);
#else
#ifdef FALCOR_INTERNAL
        float4 c = SDFVoxelCommon::calculateCubicCoefficients(rayOrigLocal, rayDirLocal, values0xx, values1xx);

        if (!rayOrigInsideVoxel) // Handle the solid side faces of the voxel.
        {
            // At this point, we know that O is located on one of the AABB faces in [0,1]^3.
            // The distance is then (Eq 4 in our sdfgrid paper): z * ( k4 + k5*x + k6*y + k7*x*y) - (k0 + k1*x + k2*y + k3*x*y)
            // But, since we have move the ray origin to the box hit, all we need to do is to evaluate f(t) = c.x * t^3 + c.y * t^2 + c.z * t + c.w, for t=0, i.e., f(0) = c.w
            if (c.w > 0.0f) // Did O hit the "solid" parts of any voxel face?
            {
                t = 0.0f;
                return t < tMax;
            }
        }

#if SCENE_SDF_VOXEL_INTERSECTION_METHOD == SCENE_SDF_VOXEL_CUBIC_SOLVER_ANALYTIC
        t = Solvers::cubicSolverAnalytic(c);
#else
        // Marmitt, NewtonRaphson, or AnalyticPlusSphereTracing
        t = Solvers::cubicSolverNumeric(c, tMax, solverMaxStepCount);
#endif
        return t >= 0.0f && t <= tMax;
#else
        return false;
#endif
#endif
    }

    /** Intersects a ray against the implicit surface surface defined by trilinearly interpolating signed distance values stored at the corner of a voxel, does not return information about the intersection.
        \param[in] rayOrigLocal The origin of the ray, in voxel local space, should be inside the voxel.
        \param[in] rayDirLocal The direction of the ray, in voxel local space.
        \param[in] rayOrigInsideVoxel True if the world ray origin is inside the voxel, note that this is note the same as rayOrigLocal.
        \param[in] values0xx The SDF corner values at x=0.
        \param[in] values1xx The SDF corner values at x=1.
        \param[in] tMax The max value of t, in voxel local space.
        \param[in] solverMaxStepCount If a numeric intersection algorithm is seleceted, this is the max number of steps that the algorithm is allowed to take.
        \return true if the ray hits the implicit surface, false otherwise.
    */
    static bool intersectSDFVoxelAny(
        const float3 rayOrigLocal,
        const float3 rayDirLocal,
        const bool rayOrigInsideVoxel,
        const float4 values0xx,
        const float4 values1xx,
        const float tMax,
        const uint solverMaxStepCount)
    {
#if SCENE_SDF_VOXEL_INTERSECTION_METHOD == SCENE_SDF_VOXEL_SPHERE_TRACING
        const float minstep = 0.0001f;
        return Solvers::sphereTraceVoxelAny(rayOrigLocal, rayDirLocal, values0xx, values1xx, minstep, tMax, solverMaxStepCount);
#else
#ifdef FALCOR_INTERNAL
        float4 c = SDFVoxelCommon::calculateCubicCoefficients(rayOrigLocal, rayDirLocal, values0xx, values1xx);

        if (!rayOrigInsideVoxel) // Handle the solid side faces of the voxel.
        {
            // At this point, we know that O is located on one of the AAABB faces in [0,1]^3.
            // The distance is then (Eq 4 in our sdfgrid paper): z* ( k4 + k5*x + k6*y + k7*x*y) - (k0 + k1*x + k2*y + k3*x*y)
            // But, since we have move the ray origin to the box hit, all we need to do is to evaluate f(t) = c.x * t^3 + c.y * t^2 + c.z * t + c.w, for t=0, i.e., f(0) = c.w
            if (c.w > 0.0f) // Did O hit the "solid" parts of any voxel face?
            {
                return  true;
            }
        }

#if SCENE_SDF_VOXEL_INTERSECTION_METHOD == SCENE_SDF_VOXEL_CUBIC_SOLVER_ANALYTIC
        return Solvers::cubicSolverAnalyticAny(c, tMax);
#else
        // Marmitt, NewtonRaphson, or AnalyticPlusSphereTracing
        return Solvers::cubicSolverNumericAny(c, tMax, solverMaxStepCount);
#endif
#else
        return false;
#endif
#endif
    }

    /** Computes the gradient at voxelUnitCoords numerically using evaluations in tetrahedron.
        Note that the vector field created by this function is discontinuous between voxels.
        \param[in] voxelUnitCoords Voxel unit coordinates, i.e., [0, 1), where the gradient should be evaluated.
        \param[in] offset The offset to use to find the tetrahedron corners with voxelUnitCoords in the center.
        \param[in] values0xx The SDF corner values at x=0.
        \param[in] values1xx The SDF corner values at x=1.
        \return the computed gradient.
    */
    static float3 computeNumericGradient(const float3 voxelUnitCoords, const float offset, const float4 values0xx, const float4 values1xx)
    {
        float2 e = float2(1.0f, -1.0f) * offset;

        // Calculate offset positions.
        float3 xyyVoxelUnitCoords = voxelUnitCoords + e.xyy;
        float3 yyxVoxelUnitCoords = voxelUnitCoords + e.yyx;
        float3 yxyVoxelUnitCoords = voxelUnitCoords + e.yxy;
        float3 xxxVoxelUnitCoords = voxelUnitCoords + e.xxx;

        float xyyD = SDFVoxelCommon::sdfVoxelTrilin(values0xx, values1xx, xyyVoxelUnitCoords);
        float yyxD = SDFVoxelCommon::sdfVoxelTrilin(values0xx, values1xx, yyxVoxelUnitCoords);
        float yxyD = SDFVoxelCommon::sdfVoxelTrilin(values0xx, values1xx, yxyVoxelUnitCoords);
        float xxxD = SDFVoxelCommon::sdfVoxelTrilin(values0xx, values1xx, xxxVoxelUnitCoords);

        return
            e.xyy * xyyD +
            e.yyx * yyxD +
            e.yxy * yxyD +
            e.xxx * xxxD;
    }

#ifdef FALCOR_INTERNAL
    /** Computes the gradient at voxelUnitCoords analytically usint trilinear interpolation.
        Note that the vector field created by this function is discontinuous between voxels.
        \param[in] voxelUnitCoords Voxel unit coordinates, i.e., [0, 1), where the gradient should be evaluated.
        \param[in] values0xx The SDF corner values at x=0.
        \param[in] values1xx The SDF corner values at x=1.
        \return the computed gradient.
    */
    static float3 computeAnalyticGradient(const float3 voxelUnitCoords, const float4 values0xx, const float4 values1xx)
    {
        float x = SDFVoxelCommon::sdfVoxelBilinCornerDiff(voxelUnitCoords.y, voxelUnitCoords.z, values1xx.x - values0xx.x, values1xx.y - values0xx.y, values1xx.z - values0xx.z, values1xx.w - values0xx.w);
        float y = SDFVoxelCommon::sdfVoxelBilinCornerDiff(voxelUnitCoords.x, voxelUnitCoords.z, values0xx.z - values0xx.x, values0xx.w - values0xx.y, values1xx.z - values1xx.x, values1xx.w - values1xx.y);
        float z = SDFVoxelCommon::sdfVoxelBilinCornerDiff(voxelUnitCoords.x, voxelUnitCoords.y, values0xx.y - values0xx.x, values0xx.w - values0xx.z, values1xx.y - values1xx.x, values1xx.w - values1xx.z);

        return float3(x, y, z);
    }
#endif

    struct Solvers
    {
        static const float kNumericSolverEpsilon = 1e-3f;

        static bool sphereTraceVoxel(
            const float3 rayOrigLocal,
            const float3 rayDirLocal,
            const float4 values0xx,
            const float4 values1xx,
            const float minstep,
            const float tMax,
            const uint maxIterations,
            out float t)
        {
            float3 p = rayOrigLocal;

            float lastD = 1.0f;
            float currD = SDFVoxelCommon::sdfVoxelTrilin(values0xx, values1xx, p);
            float clampedD = max(currD, minstep);
            t = 0.0f;

            // Check if we're already inside the surface
            if (currD <= 0.0f) return true;

            uint iterations = 0;
            for (; iterations < maxIterations; iterations++)
            {
                // Update t.
                t += clampedD;

                // Check if we're outside farplane.
                if (t > tMax)
                {
                    return false;
                }

                // Update position.
                p = clamp(rayOrigLocal + t * rayDirLocal, 0.0f, 1.0f);

                // Store lastD, and evaluate currD at current position.
                lastD = currD;
                currD = SDFVoxelCommon::sdfVoxelTrilin(values0xx, values1xx, p);
                clampedD = max(currD, minstep);

                // Check if we're inside the surface.
                if (currD <= 0.0f)
                {
                    break;
                }
            }

            // Linear interpolation to approximate intersection point.
            float finalH = currD * clampedD / (currD - lastD);
            t += finalH;

            return iterations < maxIterations;
        }

        static bool sphereTraceVoxelAny(
            const float3 rayOrigLocal,
            const float3 rayDirLocal,
            const float4 values0xx,
            const float4 values1xx,
            const float minstep,
            const float tMax,
            const uint maxIterations)
        {
            float3 p = rayOrigLocal;

            float lastD = 1.0f;
            float currD = SDFVoxelCommon::sdfVoxelTrilin(values0xx, values1xx, p);
            float clampedD = max(currD, minstep);
            float t = 0.0f;

            // Check if we're already inside the surface
            if (currD <= 0.0f) return true;

            uint iterations = 0;
            for (; iterations < maxIterations; iterations++)
            {
                // Update t.
                t += clampedD;

                // Check if we're outside farplane.
                if (t > tMax)
                {
                    return false;
                }

                // Update position.
                p = clamp(rayOrigLocal + t * rayDirLocal, 0.0f, 1.0f);

                // Store lastD, and evaluate currD at current position.
                lastD = currD;
                currD = SDFVoxelCommon::sdfVoxelTrilin(values0xx, values1xx, p);
                clampedD = max(currD, minstep);

                // Check if we're inside the surface.
                if (currD <= 0.0f)
                {
                    return true;
                }

                // Check if we're outside farplane.
                if (t > tMax)
                {
                    return false;
                }
            }

            return iterations < maxIterations;
        }

#ifdef FALCOR_INTERNAL
        // Code from David Hart: cuberoot() and solveCubicEx()
        static float cuberoot(const float x)
        {
            return abs(exp2(log2(abs(x)) * (1.0f / 3.0f))) * sign(x);
        }

        // Solve cubic equation for real roots, with a and b pre-divided by 3
        // The case of a single real root is returned three times
        static int solveCubicEx(const float a, const float b, const float c, out float3 roots)
        {
            float p = a * a - b;
            float p3 = p * p * p;
            float q = a * (p - b * 0.5f) + c * 0.5f;
            float d = q * q - p3;
            if (d >= 0.0f)
            {
                float zo2 = sqrt(d);
                float t = cuberoot(zo2 - q) + cuberoot(-zo2 - q) - a;
                roots = float3(t, t, t);
                return 1;                           // Only one root.
            }
            float v = acos(-sqrt(1.f / p3) * q) / 3.f;
            float m = cos(v);
            float n = sin(v) * sqrt(3.f);
            roots = float3(m + m, -n - m, n - m) * sqrt(p) - float3(a, a, a);
            return 3;
        }

        // Based on Jim Blinn's "How to Solve a Quadratic Equation" https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1528437
        static int quadraticSolver(const float3 coefficients, out float2 roots)
        {
            roots = {};

            const float A = coefficients.x;
            const float B = coefficients.y * 0.5f;
            const float C = coefficients.z;
            float q;
            if (B == 0.0f)
            {
                float mac = -A * C;
                if (mac < 0.0f)
                    return 0;
                q = sqrt(mac) / A;
                roots = float2(q, -q);
                return 2;
            }
            else
            {
                float b2mac = B * B - A * C;
                if (b2mac < 0.0f)
                    return 0;
                b2mac = sqrt(b2mac);
                if (B > 0.0f)
                {
                    q = B + b2mac;
                    roots = float2(-C / q, -q / A);
                }
                else
                {
                    q = -B + b2mac;
                    roots = float2(q / A, C / q);
                }
                return 2;
            }
        }

        // Returns the smallest positive root, if any. Otherwise, return a negative number.
        static float cubicSolverAnalytic(const float4 coefficients)
        {
            const float eps1 = 0.5e-3f;
            if (abs(coefficients.x) < eps1)
            {
                const float eps2 = 0.5e-3f;
                if (abs(coefficients.y) < eps2)
                {
                    // Linear equation.
                    return -coefficients.w / coefficients.z;
                }
                else
                {
                    // Second-degree equation.
                    float2 roots;
                    int numSolutions = quadraticSolver(float3(coefficients.y, coefficients.z, coefficients.w), roots);
                    if (numSolutions == 0)
                    {
                        return -1.0f;
                    }
                    else
                    {
                        float minroot = min(roots.x, roots.y);
                        return  minroot > 0.0f ? minroot : max(roots.x, roots.y);
                    }
                }
            }
            float a = coefficients.y / (coefficients.x * 3.0f);    // The solveCubicEx requires the division by 3 to be included.
            float b = coefficients.z / (coefficients.x * 3.0f);
            float c = coefficients.w / coefficients.x;

            float3 roots;
            int solutions = solveCubicEx(a, b, c, roots);
            if (solutions == 1)
            {
                return roots.x;
            }
            else
            {
                // Three solutions.
                bool foundPositiveSolution = false;
                float distance = -1.0f;
                for (int q = 0; q < 3; q++)
                {
                    if (roots[q] >= 0.0f && (!foundPositiveSolution || roots[q] < distance))
                    {
                        distance = roots[q];
                        foundPositiveSolution = true;
                    }
                }
                return distance;
            }
        }

        static bool cubicSolverAnalyticAny(const float4 coefficients, const float tMax)
        {
            float t = cubicSolverAnalytic(coefficients);
            return t >= 0.0f && t <= tMax;
        }

        static float evalCubic(const float4 coefficients, const float t)
        {
            return ((coefficients.x * t + coefficients.y) * t + coefficients.z) * t + coefficients.w;
        }

        static float evalQuadratic(const float3 coefficients, const float t)
        {
            return (coefficients.x * t + coefficients.y) * t + coefficients.z;
        }

        static float cubicSolverNumeric(const float4 coefficients, const float maxDistance, const uint solverStepCount)
        {
            float3 quadraticCoeffs = float3(3.0 * coefficients.x, 2.0 * coefficients.y, coefficients.z);
            float2 roots;
            int numRoots = quadraticSolver(quadraticCoeffs, roots);
            float t0 = 0.0, t1 = maxDistance;
            float f0 = coefficients.w;                      // This is the same as: evalCubic(coefficients, 0.0);
            float f1 = evalCubic(coefficients, t1);

            if (numRoots > 0)
            {
                float e0 = min(roots.x, roots.y);
                if (e0 >= 0.0f && e0 <= maxDistance)
                {
                    float fe0 = evalCubic(coefficients, e0);
                    // Check sign.
                    if (fe0 * f0 > 0.0f)
                    {
                        // Advance the ray to the second segment.
                        t0 = e0;
                        f0 = fe0;
                    }
                    else
                    {
                        // Use the first segment.
                        t1 = e0;
                        f1 = fe0;
                    }
                }

                float e1 = max(roots.x, roots.y);
                if (e1 >= 0.0f && e1 <= maxDistance)
                {
                    float fe1 = evalCubic(coefficients, e1);
                    // Check sign.
                    if (fe1 * f0 > 0.0f)
                    {
                        // Advance the ray to the third segment.
                        t0 = e1;
                        f0 = fe1;
                    }
                    else
                    {
                        t1 = e1;
                        f1 = fe1;
                    }
                }
            }

            // Check sign.
            if (f0 * f1 > 0.0f)
            {
                return -1.0f;
            }

            // At this point, we know there is a root in [t0, t1], and we find it with repeated linear interpolation using the f0 and f1.
            float tPrev = -FLT_MAX;
            float t = FLT_MAX;
            float ft;
#if SCENE_SDF_VOXEL_INTERSECTION_METHOD == SCENE_SDF_VOXEL_CUBIC_SOLVER_NUMERIC_MARMITT
            for (int q = 0; q < solverStepCount && abs(t - tPrev) >= kNumericSolverEpsilon; q++)
            {
                tPrev = t;
                t = (t0 * f1 - t1 * f0) / (f1 - f0);
                ft = evalCubic(coefficients, t);
                if (ft * f0 >= 0.0f)
                {
                    t0 = t;
                    f0 = ft;
                }
                else
                {
                    t1 = t;
                    f1 = ft;
                }
            }

            t = (t0 * f1 - t1 * f0) / (f1 - f0);
#elif SCENE_SDF_VOXEL_INTERSECTION_METHOD == SCENE_SDF_VOXEL_CUBIC_SOLVER_NUMERIC_NEWTON_RAPHSON
            float ft_deriv;

            // One step of linear interpolation (since we have all the data). We could also start at t=(t0+t1)*0.5, but that was worse in terms of convergence.
            tPrev = t;
            t = (t0 * f1 - t1 * f0) / (f1 - f0);
            for (int q = 0; q < solverStepCount && abs(t - tPrev) >= kNumericSolverEpsilon; q++)
            {
                ft = evalCubic(coefficients, t);
                ft_deriv = evalQuadratic(quadraticCoeffs, t);
                tPrev = t;
                t -= ft / ft_deriv;
            }
#elif SCENE_SDF_VOXEL_INTERSECTION_METHOD == SCENE_SDF_VOXEL_OPTIMIZED_SPHERE_TRACING
            tPrev = t;
            t = t0;
            for (int q = 0; q < solverStepCount && abs(t - tPrev) >= kNumericSolverEpsilon; q++)
            {
                tPrev = t;
                t -= evalCubic(coefficients, t);
            }
#endif
            return t;
        }

        static bool cubicSolverNumericAny(const float4 coefficients, const float maxDistance, const uint solverStepCount)
        {
            float3 quadraticCoeffs = float3(3.0 * coefficients.x, 2.0 * coefficients.y, coefficients.z);
            float2 roots;
            int numRoots = quadraticSolver(quadraticCoeffs, roots);
            float t0 = 0.0, t1 = maxDistance;
            float f0 = coefficients.w;                      // This is the same as: evalCubic(coefficients, 0.0);
            float f1 = evalCubic(coefficients, t1);

            if (numRoots > 0)
            {
                float e0 = min(roots.x, roots.y);
                if (e0 >= 0.0f && e0 <= maxDistance)
                {
                    float fe0 = evalCubic(coefficients, e0);
                    // Check sign.
                    if (fe0 * f0 > 0.0f)
                    {
                        // Advance the ray to the second segment.
                        t0 = e0;
                        f0 = fe0;
                    }
                    else
                    {
                        // Use the first segment.
                        t1 = e0;
                        f1 = fe0;
                    }
                }

                float e1 = max(roots.x, roots.y);
                if (e1 >= 0.0f && e1 <= maxDistance)
                {
                    float fe1 = evalCubic(coefficients, e1);
                    // Check sign.
                    if (fe1 * f0 > 0.0f)
                    {
                        // Advance the ray to the third segment.
                        t0 = e1;
                        f0 = fe1;
                    }
                    else
                    {
                        t1 = e1;
                        f1 = fe1;
                    }
                }
            }

            return f0 * f1 <= 0.0;
        }
#endif
    };
};
