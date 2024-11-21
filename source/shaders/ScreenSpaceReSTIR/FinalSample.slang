/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import Utils.Math.PackedFormats;

// Use compact packing. Changing this requires restarting the renderer.
#define USE_COMPACT_FINAL_SAMPLE 0

#if USE_COMPACT_FINAL_SAMPLE
typedef uint4 PackedFinalSample;
#else
struct PackedFinalSample
{
    float3 dir;
    float distance;
    float3 Li;
    float _pad;
};
#endif

/** Represents a final light sample used for shading pixels.
 */
struct FinalSample
{
    float3 dir;     ///< Direction to the sampled light in world-space.
    float distance; ///< Distance to the sampled light.
    float3 Li;      ///< Weighted incident radiance.

    static FinalSample createFromPacked(const PackedFinalSample packed)
    {
        FinalSample finalSample;
        finalSample.unpack(packed);
        return finalSample;
    }

    PackedFinalSample pack()
    {
        PackedFinalSample packed;
#if USE_COMPACT_FINAL_SAMPLE
        packed.x = encodeNormal2x16(dir);
        packed.y = asuint(distance);
        packed.z = encodeLogLuvHDR(Li);
        packed.w = 0;
#else
        packed.dir = dir;
        packed.distance = distance;
        packed.Li = Li;
        packed._pad = 0.f;
#endif
        return packed;
    }

    [mutating]
    void unpack(const PackedFinalSample packed)
    {
#if USE_COMPACT_FINAL_SAMPLE
        dir = decodeNormal2x16(packed.x);
        distance = asfloat(packed.y);
        Li = decodeLogLuvHDR(packed.z);
#else
        dir = packed.dir;
        distance = packed.distance;
        Li = packed.Li;
#endif
    }
};
