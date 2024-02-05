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
#error Expected FALCOR_ENABLE_TIN to be defined
#endif

// clang-format off
__glsl_extension(GL_EXT_shader_explicit_arithmetic_types)
vector<half, 2> asfloat16x2(uint a)
{
    __target_switch
    {
#if FALCOR_ENABLE_TIN
    case hlsl:
        __intrinsic_asm "asfloat16x2";
#endif
    case glsl:
        __intrinsic_asm "unpackFloat2x16";
    case spirv:
        return spirv_asm { result:$$half2 = OpBitcast $a;};
    default:
        return float16_t2(asfloat16(int16_t(a.x)), asfloat16(int16_t(a.x >> 16)));
    }
}

__glsl_extension(GL_EXT_shader_explicit_arithmetic_types)
uint asuint(vector<half, 2> a)
{
    __target_switch
    {
#if FALCOR_ENABLE_TIN
    case hlsl:
        __intrinsic_asm "asuint";
#endif
    case glsl:
        __intrinsic_asm "packFloat2x16";
    case spirv:
        return spirv_asm { result:$$uint = OpBitcast $a;};
    default:
        return (uint(asuint16(a.y)) << 16) | uint(asuint16(a.x));
    }
}
// clang-format on

#ifdef TIN_NO_FORCE_UNROLL
#define UNROLL [unroll]
#else
#define UNROLL [ForceUnroll(1024)]
#endif

namespace Tin
{
    typealias MMAMatStore = uint4;
    static const uint WAVE_SIZE = 32;

    interface IPackable
    {
        static const uint NUM_PACKED;
        associatedtype Vector : IPackableVector;
    }

    interface IPackableVector
    {
        static This unpack(uint in);
        static uint pack(This in);
    }

    interface ISharedMem
    {
        Ref<uint> get_ref(uint index);
    }

    extension half : IPackable
    {
        static const uint NUM_PACKED = 2;
        typedef half2 Vector;
    }

    extension half2 : IPackableVector
    {
        static half2 unpack(uint in)
        {
            return asfloat16x2(in);
        }

        static uint pack(half2 in)
        {
            return asuint(in);
        }
    }

    interface IPackedArray
    {
        static const uint SIZE_PACKED;
        uint get_packed_item(uint i);

        [mutating]
        void set_packed_item(uint i, uint x);
    }

    struct HPackedArray<let SIZE : uint> : IPackedArray
    {
        static const uint SIZE_PACKED = SIZE / half::NUM_PACKED;

        __subscript(uint i)->half
        {
            get {
                return val[i / half::NUM_PACKED][i % half::NUM_PACKED];
            }
            set {
                val[i / half::NUM_PACKED][i % half::NUM_PACKED] = newValue;
            }
        }

        [mutating] void fill(half2 x)
        {
            UNROLL
            for (uint i = 0; i < SIZE_PACKED; i++)
            {
                val[i] = x;
            }
        }

        [mutating]
        void clear() { fill(0.h); }

        uint get_packed_item(uint i) { return half2::pack(val[i]); }

        [mutating]
        void set_packed_item(uint i, uint x) { val[i] = half2::unpack(x); }

        half2 val[SIZE_PACKED];
    }

    // Weight offset is in terms of dwords
    static HPackedArray<Z1> mad<let Z0 : uint, let Z1 : uint>(
        HPackedArray<Z0> ip,
        HPackedArray<Z1> bias,
        StructuredBuffer<Tin::MMAMatStore> wts_buff,
        uint wt_offset = 0U
    )
    {
        HPackedArray<Z1 * 2> acc;
        acc.clear();
        uint num_mma_k = Z0 / 16;

        UNROLL
        for (uint i = 0; i < Z0 * Z1 / 8; i++)
        {
            uint4 weights = wts_buff[wt_offset / 4 + i];

            UNROLL
            for (uint j = 0; j < 4; j++)
            {
                uint linear_idx = i * 4 + j;

                uint mma_idx = linear_idx / (WAVE_SIZE * 4);
                uint sub_idx = linear_idx % (WAVE_SIZE * 4);

                uint k = (sub_idx % 2) * 4 + (sub_idx % 16) / 4;
                uint n = ((sub_idx % 4) / 2) * 8 + sub_idx / 16;

                uint k1 = k + (mma_idx % num_mma_k) * 8;
                uint n1 = n + (mma_idx / num_mma_k) * 16;

                half2 w = half2::unpack(weights[j]);

                // Get row and column
                acc.val[n1] = mad(w, ip.val[k1], acc.val[n1]);
            }
        }

        HPackedArray<Z1> res;
        UNROLL
        for (uint i = 0; i < Z1; i++)
        {
            res[i] = acc.val[i].x + acc.val[i].y + bias[i];
        }

        return res;
    };

    // Weight offset is in terms of dwords
    static HPackedArray<Z1> mad<let Z0 : uint, let Z1 : uint>(
        HPackedArray<Z0> ip,
        StructuredBuffer<Tin::MMAMatStore> wts_buff,
        uint wt_offset,
        StructuredBuffer<uint> bias_buff,
        uint bias_offset
    )
    {
        HPackedArray<Z1> bias;
        UNROLL
        for (uint i = 0; i < Z1 / 2; i++)
        {
            bias.set_packed_item(i, bias_buff[bias_offset + i]);
        }
        return mad(ip, bias, wts_buff, wt_offset);
    };

    // Weight offset is in terms of dwords
    static HPackedArray<Z1> mul<let Z0 : uint, let Z1 : uint>(
        HPackedArray<Z0> ip,
        StructuredBuffer<Tin::MMAMatStore> wts_buff,
        uint wt_offset
    )
    {
        HPackedArray<Z1> bias;
        bias.clear();
        return mad(ip, bias, wts_buff, wt_offset);
    }
};
