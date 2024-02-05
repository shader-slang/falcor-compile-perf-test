/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
import TinCommon;

namespace Tin
{
    struct HTriEncoding<let SCALE : uint>
    {
        uint width;
        uint height;

        __init(uint width, uint height)
        {
            this.width = width;
            this.height = height;
        }

        uint encode<let CH : uint>(inout HPackedArray<CH> arr, float2 uv, uint slot_offset = 0)
        {
            const float2 dim = { width, height };

            uint idx = slot_offset;
            [unroll]
            for (uint scale = SCALE; scale > 1; scale >>= 1)
            {
                float2 xy;
                half2 enc;
                xy = frac(uv * dim / scale) * 4;
                enc = abs(xy - 2) - 1;

                arr[idx] = enc.x;
                idx++;
                arr[idx] = enc.y;
                idx++;

                xy = frac(uv * dim / scale + 0.25f) * 4;
                enc = abs(xy - 2) - 1;

                arr[idx] = enc.x;
                idx++;
                arr[idx] = enc.y;
                idx++;
            }
            return idx;
        }
    }
}
