/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
namespace Tin
{
    struct HCoopVector<let SIZE : uint>
    {
        __init() {}

        __init(half init) { vec_simt.fill(init); }

        __subscript(uint i)->half
        {
            get {
                return vec_simt[i];
            }
            set {
                vec_simt[i] = newValue;
            }
        }

        HPackedArray<SIZE> vec_simt;
    };

    static HCoopVector<Z1> mad<let Z0 : uint, let Z1 : uint>(
        HCoopVector<Z0> ip, HCoopVector<Z1> bias, StructuredBuffer<Tin::MMAMatStore> wts_buff, uint wt_offset = 0U
    )
    {
        HCoopVector<Z1> res;
        res.vec_simt = mad(ip.vec_simt, bias.vec_simt, wts_buff, wt_offset);
        return res;
    };

    static HCoopVector<Z1> mad<let Z0 : uint, let Z1 : uint>(
        HCoopVector<Z0> ip, StructuredBuffer<MMAMatStore> wts_buff, uint wt_offset, StructuredBuffer<uint> bias_buff, uint bias_offset
    )
    {
        HCoopVector<Z1> res;
        res.vec_simt = mad<Z0, Z1>(ip.vec_simt, wts_buff, wt_offset, bias_buff, bias_offset);
        return res;
    };

    static HCoopVector<Z1> mul<let Z0 : uint, let Z1 : uint>(HCoopVector<Z0> ip, StructuredBuffer<MMAMatStore> wts_buff, uint wt_offset)
    {
        HCoopVector<Z1> res;
        res.vec_simt = mul<Z0, Z1>(ip.vec_simt, wts_buff, wt_offset);
        return res;
    };
};
