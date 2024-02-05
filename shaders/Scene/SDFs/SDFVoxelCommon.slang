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

import Utils.Math.FormatConversion;
import Utils.Math.AABB;

#ifdef FALCOR_INTERNAL
static const bool kEnableTightAABBsSVS = true;
static const bool kEnableTightBrickAABBsSBS = true;
static const bool kEnableTightVoxelAABBsSBS = true;
#endif

struct SDFVoxelCommon
{
    // This is the maximum level of an SDF grid. A grid cannot have a width larger than 2^kMaxLevel.
    static const uint kMaxLevel = 19;

    static const uint kLocationCodeVoxelCoordMask = (1 << kMaxLevel) - 1;
    static const uint64_t kLocationCodeVoxelCoordsMask = (1ull << (3 * kMaxLevel)) - 1;

    static const uint kLocationCodeLevelOffset = 3 * kMaxLevel;
    static const uint kLocationCodeLevelMask = 0x1f; // Equal to ceilPow2(kMaxLevel) - 1

    /** Encodes the local coordinates and level into a 64-bit location code.
        The location code is encoded as | 2 unused bits | 5 bits encoding the level | 57 bits for coordinates |.
        Note that the level is not the same as LOD, a level always represents a grid with width 2^level.
        \param[in] levelLocalVoxelCoords Voxel coordinates local to level.
        \param[in] level Which level the local coordinates are local to.
        \return A 64-bit location code stored in two 32-bit uints.
    */
    static uint2 encodeLocation(const uint3 levelLocalVoxelCoords, const uint level)
    {
        uint3 globalCoords = levelLocalToGlobalCoords(levelLocalVoxelCoords, level);
        uint64_t shiftedX = shiftCoord(globalCoords.x);
        uint64_t shiftedY = shiftCoord(globalCoords.y);
        uint64_t shiftedZ = shiftCoord(globalCoords.z);

        uint64_t locationCode = (uint64_t(level) << kLocationCodeLevelOffset) | (((shiftedX << 2) | (shiftedY << 1) | shiftedZ) & kLocationCodeVoxelCoordsMask);
        return u64to2x32(locationCode);
    }

    /** Decodes a 64-bit location code into local coordinates and a level.
        Note that the level is not the same as LOD, a level always represents a grid with width 2^level.
        \param[in] locationCode 64-bit location code.
        \param[out] levelLocalVoxelCoords Voxel coordinates local to level.
        \param[out] level Which level the local coordinates are local to.
    */
    static void decodeLocation(const uint2 locationCode, out uint3 levelLocalVoxelCoords, out uint level)
    {
        level = ((locationCode.y >> (kLocationCodeLevelOffset - 32)) & kLocationCodeLevelMask);

        uint64_t coordBits = u2x32to64(locationCode) & kLocationCodeVoxelCoordsMask;
        uint3 globalCoords = uint3(unshiftCoord(coordBits >> 2), unshiftCoord(coordBits >> 1), unshiftCoord(coordBits));
        levelLocalVoxelCoords = globalToLevelLocalCoords(globalCoords, level);
    }

    /** Creates the location code that corresponds to any of the eight children of a voxel encoded as a location code.
        \param[in] locationCode The location code of the parent voxel.
        \param[in] childID A 3-bit ID of the child for which a location code should be created, must be in the interval [0, 7] where the bit representation of childID can be interpreted as the local xyz offsets of the child.
        \param[out] childLocationCode The location code of the child.
        \return True if the child location code is valid, false otherwise.
    */
    static bool createChildLocationCode(uint2 locationCode, uint childID, out uint2 childLocationCode)
    {
        uint level = 1 + ((locationCode.y >> (kLocationCodeLevelOffset - 32)) & kLocationCodeLevelMask);
        bool valid = level < kMaxLevel;
        level = min(level, kMaxLevel);

        uint64_t bits = u2x32to64(locationCode);
        bits |= uint64_t(childID & 0x7) << (kLocationCodeLevelOffset - 3 * level);
        bits &= kLocationCodeVoxelCoordsMask;
        bits |= (uint64_t(level) << kLocationCodeLevelOffset);

        childLocationCode = u64to2x32(bits);
        return valid;
    }


    /** Shifts and dilates the low 21 bits of the value to make room for interleaving two other values of 21 bits each in the 64-bit return value.
        \param[in] x The value that should be shifted, only the low 21 bits will be kept.
        \return A 64 bit value with the highest bit unused and every third bit after that containing a bit from x.
    */
    static uint64_t shiftCoord(const uint x)
    {
        uint64_t y = uint64_t(x);
        y = (y | y << 32) & 0x1f00000000ffffull;
        y = (y | y << 16) & 0x1f0000ff0000ffull;
        y = (y | y << 8) & 0x100f00f00f00f00full;
        y = (y | y << 4) & 0x10c30c30c30c30c3ull;
        y = (y | y << 2) & 0x1249249249249249ull;
        return y;
    }

    /** Unshifts and compacts every third bits of the value to decode the 21 bit value starting at second highest bit and encompassing every third bit after that.
        \param[in] x The value that should be unshifted.
        \return A 32 bit value with the low 21 bits containing the unshifted and compacted value.
    */
    static uint unshiftCoord(const uint64_t x)
    {
        uint64_t y = x & 0x9249249249249249ull;
        y = (y | (y >> 2)) & 0x30c30c30c30c30c3ull;
        y = (y | (y >> 4)) & 0xf00f00f00f00f00full;
        y = (y | (y >> 8)) & 0x00ff0000ff0000ffull;
        y = (y | (y >> 16)) & 0xffff00000000ffffull;
        y = (y | (y >> 32)) & 0x00000000ffffffffull;
        return uint(y);
    }

    /** Converts local voxel coordinates (i.e., voxel coordinates [0, 2^level - 1]) to global voxel coordinates (i.e., [0, 2^kMaxLevel - 1]).
        \param[in] levelLocalVoxelCoords Voxel coordinates local to level.
        \param[in] level Which level the local coordinates are local to.
        \return Global voxel coordinates.
    */
    static uint3 levelLocalToGlobalCoords(const uint3 levelLocalVoxelCoords, const uint level)
    {
        return (levelLocalVoxelCoords & kLocationCodeVoxelCoordMask) << (kMaxLevel - level);
    }

    /** Converts the global voxel coordinates (i.e., voxel coordinates [0, 2^kMaxLevel - 1]) to local voxel coordinates (i.e., [0, 2^level - 1]).
        \param[in] globalCoords Global voxel coordinates.
        \param[in] level Which level the local coordinates should be local to.
        \return Local voxel coordinates, located on level.
    */
    static uint3 globalToLevelLocalCoords(const uint3 globalCoords, uint level)
    {
        return (globalCoords >> (kMaxLevel - level));
    }

#ifdef FALCOR_INTERNAL
    static void calculateVoxelUnitCoordCompEncoding(const float4 values0xx, const float4 values1xx, out uint excludedComp, out uint2 includedComps)
    {
        float xVal = abs(values0xx.x - values1xx.x) +
            abs(values0xx.y - values1xx.y) +
            abs(values0xx.z - values1xx.z) +
            abs(values0xx.w - values1xx.w);

        float yVal = abs(values0xx.x - values0xx.z) +
            abs(values0xx.y - values0xx.w) +
            abs(values1xx.x - values1xx.z) +
            abs(values1xx.y - values1xx.w);

        float zVal = abs(values0xx.x - values0xx.y) +
            abs(values0xx.z - values0xx.w) +
            abs(values1xx.x - values1xx.y) +
            abs(values1xx.z - values1xx.w);

        float maxV = max(max(xVal, yVal), zVal);

        if (maxV == xVal)
        {
            excludedComp = 0;
            includedComps = uint2(1, 2);
        }
        else if (maxV == yVal)
        {
            excludedComp = 1;
            includedComps = uint2(0, 2);
        }
        else
        {
            excludedComp = 2;
            includedComps = uint2(0, 1);
        }
    }

    static uint encodeVoxelUnitCoords(const float3 voxelUnitCoords, const float4 values0xx, const float4 values1xx, out uint excludedComp)
    {
        uint2 includedComps;
        calculateVoxelUnitCoordCompEncoding(values0xx, values1xx, excludedComp, includedComps);
        return packUnorm2x16_unsafe(float2(voxelUnitCoords[includedComps.x], voxelUnitCoords[includedComps.y]));
    }

    static uint2 calculateIncludedVoxelUnitCoordComps(const uint excludedComp)
    {
        uint2 includedComps = uint2((excludedComp + 1) % 3, (excludedComp + 2) % 3);
        return uint2(min(includedComps.x, includedComps.y), max(includedComps.x, includedComps.y));
    }

    static float3 reconstructVoxelUnitCoords(const uint excludedComp, const uint2 includedComps, const uint packedVoxelUnitCoords, const float4 values0xx, const float4 values1xx)
    {
        float4 k0_3;
        float4 k4_7;
        calculateBilinearPatchCoefficients(values0xx, values1xx, k0_3, k4_7);

        const float2 encodedVoxelUnitCoords = unpackUnorm2x16(packedVoxelUnitCoords);
        const float prod = encodedVoxelUnitCoords.x * encodedVoxelUnitCoords.y;
        float nominator;
        float denominator;

        switch (excludedComp)
        {
        case 0:
            nominator = k0_3.x + dot(float3(k0_3.z, -k4_7.x, -k4_7.z), float3(encodedVoxelUnitCoords.x, encodedVoxelUnitCoords.y, prod));
            denominator = -k0_3.y + dot(float3(-k0_3.w, k4_7.y, k4_7.w), float3(encodedVoxelUnitCoords.x, encodedVoxelUnitCoords.y, prod));
            break;
        case 1:
            nominator = k0_3.x + dot(float3(k0_3.y, -k4_7.x, -k4_7.y), float3(encodedVoxelUnitCoords.x, encodedVoxelUnitCoords.y, prod));
            denominator = -k0_3.z + dot(float3(-k0_3.w, k4_7.z, k4_7.w), float3(encodedVoxelUnitCoords.x, encodedVoxelUnitCoords.y, prod));
            break;
        case 2:
            nominator = k0_3.x + dot(k0_3.yzw, float3(encodedVoxelUnitCoords.x, encodedVoxelUnitCoords.y, prod));
            denominator = k4_7.x + dot(k4_7.yzw, float3(encodedVoxelUnitCoords.x, encodedVoxelUnitCoords.y, prod));
            break;
        default: return float3(0.0f);
        }

        float3 voxelUnitCoords;
        voxelUnitCoords[includedComps.x] = encodedVoxelUnitCoords.x;
        voxelUnitCoords[includedComps.y] = encodedVoxelUnitCoords.y;
        voxelUnitCoords[excludedComp] = nominator / denominator;
        return saturate(voxelUnitCoords);
    }

    static void calculateBilinearPatchCoefficients(const float4 values0xx, const float4 values1xx, out float4 k0_3, out float4 k4_7)
    {
        const float b = values1xx[1] - values0xx[1];
        k0_3[0] = values0xx[0];
        k0_3[1] = values1xx[0] - k0_3[0];
        k0_3[2] = values0xx[2] - k0_3[0];
        k0_3[3] = values1xx[2] - values0xx[2] - k0_3[1];
        k4_7[0] = k0_3[0] - values0xx[1];
        k4_7[1] = k0_3[1] - b;
        k4_7[2] = k0_3[2] - (values0xx[3] - values0xx[1]);
        k4_7[3] = k0_3[3] - (values1xx[3] - values0xx[3] - b);
    }

    /** Bilinearly interpolates the difference of the corners of an SDF voxel.
    */
    static float sdfVoxelBilinCornerDiff(const float x, const float y, const float diff00, const float diff01, const float diff10, const float diff11)
    {
        return lerp(lerp(diff00, diff01, y),
                    lerp(diff10, diff11, y), x);
    }

    /** Calculates the coefficients for the cubic that defines the implicit surface from the trilinearly interpolated corners of a voxel.
    */
    static float4 calculateCubicCoefficients(const float3 rayOrigLocal, const float3 rayDirLocal, const float4 values0xx, const float4 values1xx)
    {
        float4 k0_3;
        float4 k4_7;
        calculateBilinearPatchCoefficients(values0xx, values1xx, k0_3, k4_7);

        const float m0 = rayOrigLocal.x * rayOrigLocal.y;
        const float m1 = rayDirLocal.x * rayDirLocal.y;
        const float m2 = rayOrigLocal.x * rayDirLocal.y + rayOrigLocal.y * rayDirLocal.x;
        const float m3 = k4_7[1] * rayOrigLocal.z - k0_3[1];
        const float m4 = k4_7[2] * rayOrigLocal.z - k0_3[2];
        const float m5 = k4_7[3] * rayOrigLocal.z - k0_3[3];

        float4 c;
        c.x = k4_7[3] * m1 * rayDirLocal.z;
        c.y = m1 * m5 + rayDirLocal.z * (k4_7[1] * rayDirLocal.x + k4_7[2] * rayDirLocal.y + k4_7[3] * m2);
        c.z = rayDirLocal.x * m3 + rayDirLocal.y * m4 + m2 * m5 + rayDirLocal.z * (k4_7[0] + k4_7[1] * rayOrigLocal.x + k4_7[2] * rayOrigLocal.y + k4_7[3] * m0);
        c.w = k4_7[0] * rayOrigLocal.z - k0_3[0] + rayOrigLocal.x * m3 + rayOrigLocal.y * m4 + m0 * m5;

        return c;
    }
#endif // FALCOR_INTERNAL

    /** Packs the eight distance values into 8 bytes using an snorm format.
    */
    static uint2 packValues(float4 values0xx, float4 values1xx)
    {
        uint2 packedValues;
        packedValues.x = packSnorm8(values0xx[0]);
        packedValues.x |= (packSnorm8(values0xx[1]) << 8);
        packedValues.x |= (packSnorm8(values0xx[2]) << 16);
        packedValues.x |= (packSnorm8(values0xx[3]) << 24);
        packedValues.y = packSnorm8(values1xx[0]);
        packedValues.y |= (packSnorm8(values1xx[1]) << 8);
        packedValues.y |= (packSnorm8(values1xx[2]) << 16);
        packedValues.y |= (packSnorm8(values1xx[3]) << 24);
        return packedValues;
    }

    /** Unpacks eight distance values from an 8 byte snorm format.
    */
    static void unpackValues(const uint2 packedValues, out float4 values0xx, out float4 values1xx)
    {
        values0xx[0] = unpackSnorm8(packedValues.x & 0x000000ff);
        values0xx[1] = unpackSnorm8((packedValues.x >> 8) & 0x000000ff);
        values0xx[2] = unpackSnorm8((packedValues.x >> 16) & 0x000000ff);
        values0xx[3] = unpackSnorm8((packedValues.x >> 24) & 0x000000ff);
        values1xx[0] = unpackSnorm8(packedValues.y & 0x000000ff);
        values1xx[1] = unpackSnorm8((packedValues.y >> 8) & 0x000000ff);
        values1xx[2] = unpackSnorm8((packedValues.y >> 16) & 0x000000ff);
        values1xx[3] = unpackSnorm8((packedValues.y >> 24) & 0x000000ff);
    }

    /** Checks if the voxel defined by the eight corner values conservatively contains part of the implicit surface.
    */
    static bool containsSurface(const float4 values0xx, const float4 values1xx)
    {
        return  (any(values0xx <= 0.0f) || any(values1xx <= 0.0f)) &&
                (any(values0xx >= 0.0f) || any(values1xx >= 0.0f));
    }

    /** Trilinearly interpolates the eight corner values of a voxel using voxel unit coords.
    */
    static float sdfVoxelTrilin(const float4 values0xx, const float4 values1xx, const float3 voxelUnitCoords)
    {
        float4 cXs = lerp(values0xx, values1xx, voxelUnitCoords.x);
        float2 cYs = lerp(cXs.xy, cXs.zw, voxelUnitCoords.y);
        return lerp(cYs.x, cYs.y, voxelUnitCoords.z);
    }

#ifdef FALCOR_INTERNAL
    // A "local" function used by computeTightAABBFromSDFValues() below.
    static float3 _createBBoxPoint(float coord, uint axis, float remainingEdgeCoord0, float remainingEdgeCoord1)
    {
        float3 p = float3(0.0f);
        switch (axis)
        {
        case 0:
            p = float3(coord, remainingEdgeCoord0, remainingEdgeCoord1);
            break;
        case 1:
            p = float3(remainingEdgeCoord0, coord, remainingEdgeCoord1);
            break;
        case 2:
            p = float3(remainingEdgeCoord0, remainingEdgeCoord1, coord);
            break;
        }
        return p;
    }

    // A "local" function used by computeTightAABBFromSDFValues() below.
    static void _addVoxelIntersectionAlongEdgeToAABB(inout AABB box, float signedDistance0, float signedDistance1, float coord0, float coord1, uint axis, float remainingEdgeCoord0, float remainingEdgeCoord1)
    {
        bool got_zeroes = false;
        if (signedDistance0 == 0.0f)
        {
            box.include(_createBBoxPoint(coord0, axis, remainingEdgeCoord0, remainingEdgeCoord1));
            got_zeroes = true;
        }
        if (signedDistance1 == 0.0f)
        {
            box.include(_createBBoxPoint(coord1, axis, remainingEdgeCoord0, remainingEdgeCoord1));
            got_zeroes = true;
        }
        if (got_zeroes || signedDistance0 * signedDistance1 > 0.0f)
        {
            return;
        }

        float t = signedDistance0 / (signedDistance0 - signedDistance1);    // In [0,1].
        float coord =  coord0 + t * (coord1 - coord0);
        box.include(_createBBoxPoint(coord, axis, remainingEdgeCoord0, remainingEdgeCoord1));
    }

    /** Computes a tight bounding box inside a voxel based on its signed distance values at the cornes.
    \param[in] values0xx The signed distances at the voxel with x=0 (in a unit box).
    \param[in] values1xx The signed distances at the voxel with x=1 (in a unit box).
    \param[in] voxelAABB The original voxel box.
    \return A thight AABB.
    */
    static AABB computeTightAABBFromSDFValues(float4 values0xx, float4 values1xx, AABB voxelAABB)
    {
        AABB tightBox;
        tightBox.invalidate();

        // Four edges in the x-direction.
        _addVoxelIntersectionAlongEdgeToAABB(tightBox, values0xx.x, values1xx.x, voxelAABB.minPoint.x, voxelAABB.maxPoint.x, 0, voxelAABB.minPoint.y, voxelAABB.minPoint.z);    // Test edge in x with y=-1, z=-1.
        _addVoxelIntersectionAlongEdgeToAABB(tightBox, values0xx.z, values1xx.z, voxelAABB.minPoint.x, voxelAABB.maxPoint.x, 0, voxelAABB.maxPoint.y, voxelAABB.minPoint.z);    // Test edge in x with y=+1, z=-1.
        _addVoxelIntersectionAlongEdgeToAABB(tightBox, values0xx.y, values1xx.y, voxelAABB.minPoint.x, voxelAABB.maxPoint.x, 0, voxelAABB.minPoint.y, voxelAABB.maxPoint.z);    // Test edge in x with y=-1, z=+1.
        _addVoxelIntersectionAlongEdgeToAABB(tightBox, values0xx.w, values1xx.w, voxelAABB.minPoint.x, voxelAABB.maxPoint.x, 0, voxelAABB.maxPoint.y, voxelAABB.maxPoint.z);    // Test edge in x with y=+1, z=+1.
        // Four edges in the y-direction.
        _addVoxelIntersectionAlongEdgeToAABB(tightBox, values0xx.x, values0xx.z, voxelAABB.minPoint.y, voxelAABB.maxPoint.y, 1, voxelAABB.minPoint.x, voxelAABB.minPoint.z);    // Test edge in y with x=-1, z=-1.
        _addVoxelIntersectionAlongEdgeToAABB(tightBox, values1xx.x, values1xx.z, voxelAABB.minPoint.y, voxelAABB.maxPoint.y, 1, voxelAABB.maxPoint.x, voxelAABB.minPoint.z);    // Test edge in y with x=+1, z=-1.
        _addVoxelIntersectionAlongEdgeToAABB(tightBox, values0xx.y, values0xx.w, voxelAABB.minPoint.y, voxelAABB.maxPoint.y, 1, voxelAABB.minPoint.x, voxelAABB.maxPoint.z);    // Test edge in y with x=-1, z=+1.
        _addVoxelIntersectionAlongEdgeToAABB(tightBox, values1xx.y, values1xx.w, voxelAABB.minPoint.y, voxelAABB.maxPoint.y, 1, voxelAABB.maxPoint.x, voxelAABB.maxPoint.z);    // Test edge in y with x=+1, z=+1.
        // Four edges in the z-direction.
        _addVoxelIntersectionAlongEdgeToAABB(tightBox, values0xx.x, values0xx.y, voxelAABB.minPoint.z, voxelAABB.maxPoint.z, 2, voxelAABB.minPoint.x, voxelAABB.minPoint.y);    // Test edge in z with x=-1, y=-1.
        _addVoxelIntersectionAlongEdgeToAABB(tightBox, values1xx.x, values1xx.y, voxelAABB.minPoint.z, voxelAABB.maxPoint.z, 2, voxelAABB.maxPoint.x, voxelAABB.minPoint.y);    // Test edge in z with x=+1, y=-1.
        _addVoxelIntersectionAlongEdgeToAABB(tightBox, values0xx.z, values0xx.w, voxelAABB.minPoint.z, voxelAABB.maxPoint.z, 2, voxelAABB.minPoint.x, voxelAABB.maxPoint.y);    // Test edge in z with x=-1, y=+1.
        _addVoxelIntersectionAlongEdgeToAABB(tightBox, values1xx.z, values1xx.w, voxelAABB.minPoint.z, voxelAABB.maxPoint.z, 2, voxelAABB.maxPoint.x, voxelAABB.maxPoint.y);    // Test edge in z with x=+1, y=+1.

        const float eps = 1.0e-7;
        if (tightBox.minPoint.x == tightBox.maxPoint.x)
        {
            tightBox.minPoint.x = max(tightBox.minPoint.x - eps, voxelAABB.minPoint.x);
            tightBox.maxPoint.x = min(tightBox.maxPoint.x + eps, voxelAABB.maxPoint.x);
        }
        if (tightBox.minPoint.y == tightBox.maxPoint.y)
        {
            tightBox.minPoint.y = max(tightBox.minPoint.y - eps, voxelAABB.minPoint.y);
            tightBox.maxPoint.y = min(tightBox.maxPoint.y + eps, voxelAABB.maxPoint.y);
        }
        if (tightBox.minPoint.z == tightBox.maxPoint.z)
        {
            tightBox.minPoint.z = max(tightBox.minPoint.z - eps, voxelAABB.minPoint.z);
            tightBox.maxPoint.z = min(tightBox.maxPoint.z + eps, voxelAABB.maxPoint.z);
        }
        return tightBox;
    }
#endif // FALCOR_INTERNAL

};
