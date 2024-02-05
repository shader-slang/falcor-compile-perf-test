/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
#if !FALCOR_ENABLE_TIN || !defined(FALCOR_D3D12)
#error This module requires FALCOR_D3D12 and FALCOR_ENABLE_TIN
#endif

namespace Tin
{
    struct HCoopVector<let SIZE : uint>
    {
        __init() { dirty = false; }

        __init(half init)
        {
            dirty = false;
            vec_simt.fill(init);
            vec_warp.fill(init);
        }

        __subscript(uint i)->half
        {
            get {
                return vec_simt[i];
            }
            set {
                vec_simt[i] = newValue;
                dirty = true;
            }
        }

        HPackedArray<SIZE> vec_simt;
        bool dirty;
        HVector<SIZE> vec_warp;
    };

    static HCoopVector<Z1> mad<let Z0 : uint, let Z1 : uint>(
        HCoopVector<Z0> ip, HCoopVector<Z1> bias, StructuredBuffer<Tin::MMAMatStore> wts_buff, uint wt_offset = 0U
    )
    {
        HCoopVector<Z1> res;
        bool uniform = (WaveActiveCountBits(true) == WaveGetLaneCount()) && WaveActiveAllEqual(wt_offset);

        if (uniform)
        {
            if (WaveActiveAnyTrue(ip.dirty))
            {
                ip.vec_warp.from_array(ip.vec_simt);
            }

            if (WaveActiveAnyTrue(bias.dirty))
            {
                bias.vec_warp.from_array(bias.vec_simt);
            }

            res.vec_warp = mad(ip.vec_warp, bias.vec_warp, wts_buff, wt_offset);
            res.vec_warp.to_array(res.vec_simt);
        }
        else
        {
            res.vec_simt = mad(ip.vec_simt, bias.vec_simt, wts_buff, wt_offset);
        }
        res.dirty = false;
        return res;
    };

    static HCoopVector<Z1> mad<let Z0 : uint, let Z1 : uint>(
        HCoopVector<Z0> ip, StructuredBuffer<MMAMatStore> wts_buff, uint wt_offset, StructuredBuffer<uint> bias_buff, uint bias_offset
    )
    {
        HCoopVector<Z1> res;
        bool uniform =
            (WaveActiveCountBits(true) == WaveGetLaneCount()) && WaveActiveAllEqual(wt_offset) && WaveActiveAllEqual(bias_offset);

        if (uniform)
        {
            if (WaveActiveAnyTrue(ip.dirty))
            {
                ip.vec_warp.from_array(ip.vec_simt);
            }

            res.vec_warp = mad<Z0, Z1>(ip.vec_warp, wts_buff, wt_offset, bias_buff, bias_offset);
            res.vec_warp.to_array(res.vec_simt);
        }
        else
        {
            res.vec_simt = mad<Z0, Z1>(ip.vec_simt, wts_buff, wt_offset, bias_buff, bias_offset);
        }
        res.dirty = false;
        return res;
    };

    static HCoopVector<Z1> mul<let Z0 : uint, let Z1 : uint>(HCoopVector<Z0> ip, StructuredBuffer<MMAMatStore> wts_buff, uint wt_offset)
    {
        HCoopVector<Z1> res;
        bool uniform = (WaveActiveCountBits(true) == WaveGetLaneCount()) && WaveActiveAllEqual(wt_offset);

        if (uniform)
        {
            if (WaveActiveAnyTrue(ip.dirty))
            {
                ip.vec_warp.from_array(ip.vec_simt);
            }

            res.vec_warp = mul<Z0, Z1>(ip.vec_warp, wts_buff, wt_offset);
            res.vec_warp.to_array(res.vec_simt);
        }
        else
        {
            res.vec_simt = mul<Z0, Z1>(ip.vec_simt, wts_buff, wt_offset);
        }
        res.dirty = false;
        return res;
    };
};
