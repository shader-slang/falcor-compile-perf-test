/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
#ifndef FALCOR_ENABLE_TIN
#error FALCOR_ENABLE_TIN is not defined!
#endif

import Internal.Utils.Neural.TIN.TinCommon;

/**
 * The network parameters are stored in a StructuredBuffer in order to
 * enable vectorized loads (StructuredBuffer loads are aligned by the
 * struct stride, unlike ByteAddressBuffer).
 *
 * There are several considerations to make this optimal:
 * 1. The native data type loaded should be as wide as possible.
 * Therefore we use 16B types for FP32 weights and 8B types for fp16.
 * 2. Conversion from integer to fp16 types is poorly optimized.
 * Hence we have to use fp32 or fp16 explicitly in the struct.
 * 3. Loads from arrays of native types are poorly optimized to vectorized
 * loads, even when array size and indexing are statically known.
 *
 * For these reasons, there is one struct type for storing parameters for
 * networks running in fp32 precision, and one type for fp16 precision.
 * The get*() functions are written to enable the best code generation.
 */

/**
 * Storage type for network parameters in FP16 format.
 * This is a 32B binary blob for vectorized loads.
 */
struct NetworkParamStorageFP16
{
#if !FALCOR_ENABLE_TIN
    // This implementation generates LDG.64 instructions.
    float16_t4 a, b, c, d;

    float16_t2 getFloat16_2(uint i)
    {
        switch (i)
        {
        case 0:
            return a.xy;
        case 1:
            return a.zw;
        case 2:
            return b.xy;
        case 3:
            return b.zw;
        case 4:
            return c.xy;
        case 5:
            return c.zw;
        case 6:
            return d.xy;
        case 7:
            return d.zw;
        }
        return {};
    }

    // This accessor is unoptimized.
    // It may need unrolling similar to above for best code generation.
    float16_t getFloat16(uint i)
    {
        if (i < 4)
            return a[i];
        else if (i < 8)
            return b[i - 4];
        else if (i < 12)
            return c[i - 8];
        else
            return d[i - 12];
    }

#else
    // This implementation generates LDG.128 instructions on Ampere and newer.
    uint4 a, b;

    float16_t2 getFloat16_2(uint i)
    {
        switch (i)
        {
        case 0:
            return asfloat16x2(a.x);
        case 1:
            return asfloat16x2(a.y);
        case 2:
            return asfloat16x2(a.z);
        case 3:
            return asfloat16x2(a.w);
        case 4:
            return asfloat16x2(b.x);
        case 5:
            return asfloat16x2(b.y);
        case 6:
            return asfloat16x2(b.z);
        case 7:
            return asfloat16x2(b.w);
        }
        return {};
    }

    float16_t getFloat16(uint i)
    {
        if (i < 8)
        {
            float16_t2 f = asfloat16x2(a[i >> 1]);
            return f[i & 1];
        }
        else
        {
            float16_t2 f = asfloat16x2(b[(i - 8) >> 1]);
            return f[i & 1];
        }
    }
#endif // FALCOR_ENABLE_TIN
};

/**
 * Storage type for network parameters in FP32 format.
 * This is a 32B binary blob for vectorized loads.
 */
struct NetworkParamStorageFP32
{
    // Note that this must be written as two float4 for the compiler to fully optimize the loads.
    // Arrays of float of float4 types sometime fail to optimize to vectorized loads.
    // Also matrix types float2x4 fail to vectorize.
    float4 a, b;

    float getFloat32(uint i) { return i < 4 ? a[i] : b[i - 4]; }
};
