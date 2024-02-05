/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/

/**
 * This file contains utilities for generating low-discrepancy point sets.
 *
 * Note: This code should NOT be released publicly due to NVIDIA IP.
 */

/**
 * Generates 2D coords for point j out of 2^m points for a 3D (t,m,s)-net with good 2D projections.
 * The point set projects to the Larcher-Pillichshammer points in 2D and is indexed by j in the 3rd dimension.
 * See details in Sec. 5 of Munkberg et al., "Hierarchical stochastic motion blur rasterization", 2011.
 * https://dl.acm.org/citation.cfm?id=2018341.
 *
 * @param[in] j Point index in the range 0 <= j < 2^m.
 * @param[in] m The size of the point set is 2^m, where m < 32.
 * @return 2D coordinates in [0,1) for m <= 24. Note for m > 24 the range is inclusive on both sides due to floating-point precision.
 */
float2 generatePermutedTms(uint j, uint m)
{
    uint x = 0, y = 0;
    uint c1 = 3, c2 = 1 << (m - 1);
    for (j <<= (32 - m); j != 0; j <<= 1)
    {
        if (j & (1 << 31))
        {
            x ^= (c1 >> 1);
            y ^= c2;
        }
        c1 ^= (c1 << 1);
        c2 ^= (c2 >> 1);
    }

    // x,y are in 0.m bits fixed point format
    return float2(x, y) * (1.f / (1 << m)); // Scale by 2^-m
}
