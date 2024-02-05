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

import Tin;
import Internal.Utils.Neural.ActivationFunctions;

namespace Tin
{
    typealias RWWeightBuff = RWStructuredBuffer<Tin::MMAMatStore>;
    typealias WeightBuff = StructuredBuffer<Tin::MMAMatStore>;
    typealias BiasBuff = StructuredBuffer<uint>;

    struct HLinear<let Z0 : uint, let Z1 : uint>
    {
        typealias WtMatrix = HMatrixB<Z0, Z1>;
        typealias OutVector = HVector<Z1>;

        static HVector<Z1> forward(HVector<Z0> ip, WeightBuff wts_buff, BiasBuff bias_buff, uint wt_offset = 0U, uint bias_offset = 0U)
        {
            WtMatrix wt;
            wt.load_mma(wts_buff, wt_offset);

            OutVector bias;
            bias.load_uniform(bias_buff, bias_offset);

            OutVector out;
            out = mad(ip, wt, bias);
            return out;
        }

        static uint weight_offset() { return Z0 * Z1 / half::NUM_PACKED; }

        static uint bias_offset() { return Z1 / half::NUM_PACKED; }
    }

    struct HMLP<
        let HIDDEN_LAYERS : uint,
        let Z_INPUT : uint,
        let Z_HIDDEN : uint,
        let Z_OUTPUT : uint,
        HIDDEN_ACT : IHActFn,
        OUTPUT_ACT : IHActFn>
    {
        uint wt_offset;
        uint bias_offset;

        HLinear<Z_INPUT, Z_HIDDEN> ip_layer;
        HLinear<Z_HIDDEN, Z_HIDDEN> hidden_layer;
        HLinear<Z_HIDDEN, Z_OUTPUT> op_layer;

        [mutating]
        HVector<Z_OUTPUT> forward(
            HVector<Z_INPUT> ip,
            WeightBuff wts_buff,
            BiasBuff bias_buff,
            uint wt_base_offset = 0U,
            uint g_tin_mlp_bias_offset = 0U
        )
        {
            wt_offset = wt_base_offset;
            bias_offset = g_tin_mlp_bias_offset;

            var h = ip_layer.forward(ip, wts_buff, bias_buff, wt_offset, bias_offset);
            wt_offset += ip_layer.weight_offset();
            bias_offset += ip_layer.bias_offset();

            HIDDEN_ACT h_act;
            h = happly(h_act, h);

            [unroll]
            for (uint i = 0; i < HIDDEN_LAYERS; i++)
            {
                h = hidden_layer.forward(h, wts_buff, bias_buff, wt_offset, bias_offset);
                wt_offset += hidden_layer.weight_offset();
                bias_offset += hidden_layer.bias_offset();
                h = happly(h_act, h);
            }

            OUTPUT_ACT op_act;

            var op = op_layer.forward(h, wts_buff, bias_buff, wt_offset, bias_offset);
            op = happly(op_act, op);
            return op;
        }
    }
}
