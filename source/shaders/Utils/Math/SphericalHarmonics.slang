/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import Utils.Math.MathHelpers;

/**
 * Get sequential index of SH basis function of degree l>=0 and order m in [-l,l].
 */
uint get_SH_index(int l, int m)
{
    // There are l^2 basis functions of degree 0..l-1. And we add +l to shift order to positive values.
    return (uint)(l * (l + 1) + m);
}

/**
 * Evaluates the spherical harmonics basis function Y_i at Cartesian coordinate p=(x,y,z) on the unit sphere.
 * @param[in] idx Sequential SH basis function index, where 0 <= idx < 16 (SH degree 3 and lower).
 * @param[in] p Cartesian coordinate p=(x,y,z) on the unit sphere.
 * @return Evaluated SH basis function.
 */
float eval_SH(uint idx, float3 p)
{
    // Standard real SH basis. See https://en.wikipedia.org/wiki/Table_of_spherical_harmonics
    // Note that in Appendix A2 of Sloan 2008, "Stupid Spherical Harmonics (SH) Tricks",
    // the signs are reversed for basis functions with odd m. We're not using Sloan's definitions.
    // TODO: More general implementation that supports higher degrees.
    // clang-format off
    switch (idx)
    {
    // l = 0
    case 0: return 1.f / (2.f * M_SQRT_PIf);                // m = 0
    // l = 1
    case 1: return p.y * sqrt(3.f) / (2.f * M_SQRT_PIf);            // m =-1
    case 2: return p.z * sqrt(3.f) / (2.f * M_SQRT_PIf);            // m = 0
    case 3: return p.x * sqrt(3.f) / (2.f * M_SQRT_PIf);            // m = 1
    // l = 2
    case 4: return p.x * p.y * sqrt(15.f) / (2.f * M_SQRT_PIf);                 // m =-2
    case 5: return p.y * p.z * sqrt(15.f) / (2.f * M_SQRT_PIf);                 // m =-1
    case 6: return (3.f * p.z * p.z - 1.f) * sqrt(5.f) / (4.f * M_SQRT_PIf);    // m = 0
    case 7: return p.x * p.z * sqrt(15.f) / (2.f * M_SQRT_PIf);                 // m = 1
    case 8: return (p.x * p.x - p.y * p.y) * sqrt(15.f) / (4.f * M_SQRT_PIf);   // m = 2
    // l = 3
    case 9:  return p.y * (3.f * p.x * p.x - p.y * p.y) * sqrt(70.f) / (8.f * M_SQRT_PIf);  // m =-3
    case 10: return p.x * p.y * p.z * sqrt(105.f) / (2.f * M_SQRT_PIf);                     // m =-2
    case 11: return p.y * (5.f * p.z * p.z - 1.f) * sqrt(42.f) / (8.f * M_SQRT_PIf);        // m =-1
    case 12: return p.z * (5.f * p.z * p.z - 3.f) * sqrt(7.f) / (4.f * M_SQRT_PIf);         // m = 0
    case 13: return p.x * (5.f * p.z * p.z - 1.f) * sqrt(42.f) / (8.f * M_SQRT_PIf);        // m = 1
    case 14: return p.z * (p.x * p.x - p.y * p.y) * sqrt(105.f) / (4.f * M_SQRT_PIf);       // m = 2
    case 15: return p.x * (p.x * p.x - 3.f * p.y * p.y) * sqrt(70.f) / (8.f * M_SQRT_PIf);  // m = 3
    }
    // clang-format on
    return 0.f;
}

/**
 * Evaluates the spherical harmonics basis function Y_l^m at Cartesian coordinate p=(x,y,z) on the unit sphere.
 * @param[in] l SH degree 0 <= l < 3.
 * @param[in] m SH order m in [-l,l].
 * @param[in] p Cartesian coordinate p=(x,y,z) on the unit sphere.
 * @return Evaluated SH basis function.
 */
float eval_SH(int l, int m, float3 p)
{
    return eval_SH(get_SH_index(l, m), p);
}
