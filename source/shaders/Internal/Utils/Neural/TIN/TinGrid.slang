/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
#if FALCOR_ENABLE_TIN && defined(FALCOR_D3D12)
import Tin;
#else
import TinCommon;
#endif

namespace Tin
{
    struct HFeatureGrid<let FEATURE_CH : uint>
    {
        uint width;
        uint height;

        __init(uint width, uint height)
        {
            this.width = width;
            this.height = height;
        }

        half2 blend(half2 x0, half2 x1, half a) { return x0 * (1.h - a) + x1 * a; }

        void sample<let CH : uint>(
            inout Tin::HPackedArray<CH> arr,
            StructuredBuffer<uint> buffer,
            float2 uv,
            uint buffer_offset = 0,
            uint slot_offset = 0
        )
        {
            float2 xy = uv * float2(width, height) - 0.5f;
            float2 xy_base = floor(xy);
            float2 alpha = xy - xy_base;

            float x0 = max(0.f, xy_base.x);
            float y0 = max(0.f, xy_base.y);
            float x1 = min(width - 1, xy_base.x + 1);
            float y1 = min(height - 1, xy_base.y + 1);

            const uint NUM_CHUNKS = FEATURE_CH / 2;

            uint baseIndex00 = (y0 * width + x0) + buffer_offset;
            uint baseIndex01 = (y0 * width + x1) + buffer_offset;
            uint baseIndex10 = (y1 * width + x0) + buffer_offset;
            uint baseIndex11 = (y1 * width + x1) + buffer_offset;

            uint idx = slot_offset;
            [unroll]
            for (uint i = 0; i < NUM_CHUNKS; i++)
            {
                uint chunk00 = buffer[baseIndex00 + i * width * height];
                uint chunk01 = buffer[baseIndex01 + i * width * height];
                uint chunk10 = buffer[baseIndex10 + i * width * height];
                uint chunk11 = buffer[baseIndex11 + i * width * height];

                half2 d00 = half2::unpack(chunk00);
                half2 d01 = half2::unpack(chunk01);
                half2 d10 = half2::unpack(chunk10);
                half2 d11 = half2::unpack(chunk11);

                half2 d = blend(blend(d00, d01, alpha.x), blend(d10, d11, alpha.x), alpha.y);
                arr[idx] = d.x;
                idx++;
                arr[idx] = d.y;
                idx++;
            }
        }

#if FALCOR_ENABLE_TIN && defined(FALCOR_D3D12)
        // This function currently requires TIN due to the definition of atomic_add_half2() being in the Tin module.
        // TODO: Move atomic_add_half2 out of Tin as it only depends on NVAPI being available.
        void backward<let CH : uint>(
            Tin::HPackedArray<CH> grad_arr,
            RWByteAddressBuffer buffer,
            float2 uv,
            uint buffer_offset = 0,
            uint slot_offset = 0
        )
        {
            float2 xy = uv * float2(width, height) - 0.5f;
            float2 xy_base = floor(xy);
            float2 alpha = xy - xy_base;
            float2 alpha_n = 1 - alpha;

            float w00 = alpha_n.x * alpha_n.y;
            float w01 = alpha.x * alpha_n.y;
            float w10 = alpha_n.x * alpha.y;
            float w11 = alpha.x * alpha.y;

            float x0 = max(0.f, xy_base.x);
            float y0 = max(0.f, xy_base.y);
            float x1 = min(width - 1, xy_base.x + 1);
            float y1 = min(height - 1, xy_base.y + 1);

            const uint NUM_CHUNKS = FEATURE_CH / 2;

            uint baseIndex00 = (y0 * width + x0) + buffer_offset;
            uint baseIndex01 = (y0 * width + x1) + buffer_offset;
            uint baseIndex10 = (y1 * width + x0) + buffer_offset;
            uint baseIndex11 = (y1 * width + x1) + buffer_offset;

            [unroll]
            for (uint i = 0; i < NUM_CHUNKS; i++)
            {
                uint addr00 = baseIndex00 + i * width * height;
                uint addr01 = baseIndex01 + i * width * height;
                uint addr11 = baseIndex11 + i * width * height;
                uint addr10 = baseIndex10 + i * width * height;

                half2 gradh;
                gradh.x = grad_arr[slot_offset + i * 2];
                gradh.y = grad_arr[slot_offset + i * 2 + 1];

                atomic_add_half2(buffer, addr00 * 4, half2::pack(gradh * w00));
                atomic_add_half2(buffer, addr01 * 4, half2::pack(gradh * w01));
                atomic_add_half2(buffer, addr10 * 4, half2::pack(gradh * w10));
                atomic_add_half2(buffer, addr11 * 4, half2::pack(gradh * w11));
            }
        }
#endif
    }
}
