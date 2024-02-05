/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
#ifndef FALCOR_ENABLE_NV_COOP_VECTOR
#error Expected FALCOR_ENABLE_NV_COOP_VECTOR to be defined
#endif

#if !FALCOR_ENABLE_NV_COOP_VECTOR || !defined(FALCOR_VULKAN)
#error This module requires FALCOR_VULKAN and FALCOR_ENABLE_NV_COOP_VECTOR
#endif

namespace Tin
{
    struct HCoopVector<let SIZE : uint>
    {
        __init() {}

        __init(float16_t init) { vec_simt = coopvecNV<float16_t, SIZE>(init); }

        __subscript(uint i)->half
        {
            get {
                return vec_simt[i];
            }
            set {
                vec_simt[i] = newValue;
            }
        }

        coopvecNV<float16_t, SIZE> vec_simt;
    }

    static HCoopVector<Z1> mad<let Z0 : uint, let Z1 : uint>(
        HCoopVector<Z0> ip, StructuredBuffer<float16_t> wts_buff, uint wt_offset, StructuredBuffer<float16_t> bias_buff, uint bias_offset
    )
    {
        const int wtLayout = gl_CooperativeVectorMatrixLayoutInferencingOptimalNV;
        const int ipInterpret = gl_ComponentTypeFloat16NV;
        const int wtInterpret = gl_ComponentTypeFloat16NV;
        const int biasInterpret = gl_ComponentTypeFloat16NV;
        const int stride = 0;

        HCoopVector<Z1> res;
        coopVecMatMulAddNV<float16_t, Z1, float16_t, Z0, float16_t>(
            res.vec_simt,
            ip.vec_simt,
            ipInterpret,
            wts_buff,
            wt_offset * 4,
            wtInterpret,
            bias_buff,
            bias_offset * 4,
            biasInterpret,
            Z1,
            Z0,
            wtLayout,
            false,
            stride
        );
        return res;
    };

    static HCoopVector<Z1> mul<let Z0 : uint, let Z1 : uint>(HCoopVector<Z0> ip, StructuredBuffer<float16_t> wts_buff, uint wt_offset)
    {
        const int wtLayout = gl_CooperativeVectorMatrixLayoutInferencingOptimalNV;
        const int ipInterpret = gl_ComponentTypeFloat16NV;
        const int wtInterpret = gl_ComponentTypeFloat16NV;
        const int stride = 0;

        HCoopVector<Z1> res;
        coopVecMatMulNV<float16_t, Z1, float16_t, Z0, float16_t>(
            res.vec_simt, ip.vec_simt, ipInterpret, wts_buff, wt_offset * 4, wtInterpret, Z1, Z0, wtLayout, false, stride
        );
        return res;
    };
};

// Arithmetic operators
static Tin::HCoopVector<Z> operator +<let Z : uint>(Tin::HCoopVector<Z> a, Tin::HCoopVector<Z> b)
{
    Tin::HCoopVector<Z> r;
    r.vec_simt = a.vec_simt + b.vec_simt;
    return r;
}

static Tin::HCoopVector<Z> operator -<let Z : uint>(Tin::HCoopVector<Z> a, Tin::HCoopVector<Z> b)
{
    Tin::HCoopVector<Z> r;
    r.vec_simt = a.vec_simt - b.vec_simt;
    return r;
}

Tin::HCoopVector<Z> operator*<let Z : uint>(Tin::HCoopVector<Z> a, Tin::HCoopVector<Z> b)
{
    Tin::HCoopVector<Z> r;
    r.vec_simt = a.vec_simt * b.vec_simt;
    return r;
}

Tin::HCoopVector<Z> operator /<let Z : uint>(Tin::HCoopVector<Z> a, Tin::HCoopVector<Z> b)
{
    Tin::HCoopVector<Z> r;
    r.vec_simt = a.vec_simt / b.vec_simt;
    return r;
}

// Misc math functions
Tin::HCoopVector<Z> min<let Z : uint>(Tin::HCoopVector<Z> a, Tin::HCoopVector<Z> b)
{
    Tin::HCoopVector<Z> r;
    r.vec_simt = min(a.vec_simt, b.vec_simt);
    return r;
}

Tin::HCoopVector<Z> max<let Z : uint>(Tin::HCoopVector<Z> a, Tin::HCoopVector<Z> b)
{
    Tin::HCoopVector<Z> r;
    r.vec_simt = max(a.vec_simt, b.vec_simt);
    return r;
}

Tin::HCoopVector<Z> exp<let Z : uint>(Tin::HCoopVector<Z> a)
{
    Tin::HCoopVector<Z> r;
    r.vec_simt = exp(a.vec_simt);
    return r;
}
