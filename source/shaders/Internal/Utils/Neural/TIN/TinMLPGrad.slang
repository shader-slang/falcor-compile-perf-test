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
    typealias WeightBuff = StructuredBuffer<Tin::MMAMatStore>;
    typealias RWWeightBuff = RWStructuredBuffer<Tin::MMAMatStore>;
    typealias BiasBuff = StructuredBuffer<uint>;
    typealias RWGradBuff = RWByteAddressBuffer;

    struct HLinearGrad<let Z0 : uint, let Z1 : uint, let WARPS_PER_WG : uint, let WEIGHT_SIZE : uint>
    {
        static const uint WEIGHT_SMEM_SIZE = WARPS_PER_WG == 1 ? WEIGHT_SIZE : WARPS_PER_WG * WEIGHT_SIZE / 2;

        static HVector<Z1> forward(HVector<Z0> ip, WeightBuff wts_buff, BiasBuff bias_buff, uint wt_offset = 0U, uint bias_offset = 0U)
        {
            HMatrixB<Z0, Z1> wt;
            wt.load_mma(wts_buff, wt_offset);

            HVector<Z1> bias;
            bias.load_uniform(bias_buff, bias_offset);

            return mad(ip, wt, bias);
        }

        static void reduce_bias(HVector<Z1> bd, RWGradBuff bias_grad, bool accumulate, uint bias_offset, uint warp_id)
        {
            // Reduce inside warp
            const uint COLS_PACKED = Z1 / half::NUM_PACKED;
            reduce_sum<Z1, WEIGHT_SMEM_SIZE>(bd, warp_id * COLS_PACKED);
            AllMemoryBarrierWithGroupSync();

            // Reduce across warp
            const uint NUM_VECTORS = WARPS_PER_WG * WAVE_SIZE;
            const uint COLS_PER_THREAD = COLS_PACKED / NUM_VECTORS;
            const uint REDUCTION_ITERS = (COLS_PACKED % NUM_VECTORS > 0) ? COLS_PER_THREAD + 1 : COLS_PER_THREAD;

            uint lane_id = WaveGetLaneIndex();
            uint packed_col_idx = warp_id * WAVE_SIZE + lane_id;

            [unroll]
            for (uint j = 0; j < REDUCTION_ITERS; j++)
            {
                if (packed_col_idx < COLS_PACKED)
                {
                    half2 red_reg = half2(0.f);

                    [unroll]
                    for (uint i = 0; i < WARPS_PER_WG; i++)
                    {
                        uint shared_data = tinSharedGrad<WEIGHT_SMEM_SIZE>(packed_col_idx + i * COLS_PACKED);
                        red_reg += half2::unpack(shared_data);
                    }

                    uint red_regu = half2::pack(red_reg);
                    uint offset = (bias_offset + packed_col_idx) * 4;

                    if (accumulate)
                        atomic_add_half2(bias_grad, offset, red_regu);
                    else
                        bias_grad.Store(offset, red_regu);
                }

                packed_col_idx += NUM_VECTORS;
            }
            AllMemoryBarrierWithGroupSync();
        }

        static void reduce_weights(HMatrixB<Z0, Z1> wd, RWGradBuff wts_grad, bool accumulate, uint wt_offset, uint warp_id)
        {
            [unroll]
            for (uint j = 2; j <= WARPS_PER_WG; j <<= 1)
            {
                if (warp_id % j == j / 2)
                {
                    wd.store_mma<WEIGHT_SMEM_SIZE>((warp_id / j) * WEIGHT_SIZE);
                }
                AllMemoryBarrierWithGroupSync();

                if (warp_id % j == 0)
                {
                    HMatrixB<Z0, Z1> wd_1;
                    wd_1.load_mma<WEIGHT_SMEM_SIZE>((warp_id / j) * WEIGHT_SIZE);
                    wd = wd + wd_1;
                }
                AllMemoryBarrierWithGroupSync();
            }

            if (warp_id == 0)
            {
                wd.store_mma<WEIGHT_SMEM_SIZE>(0);

                hload_and_accumulate_mma<Z0, Z1, WEIGHT_SMEM_SIZE>(wts_grad, 0, wt_offset, accumulate);
            }
            AllMemoryBarrierWithGroupSync();
        }

        static HVector<Z0> backward(
            HVector<Z0> ip,
            HVector<Z1> grad,
            WeightBuff wts_buff,
            RWGradBuff wts_grad,
            RWGradBuff bias_grad,
            bool accumulate,
            uint wt_offset,
            uint bias_offset,
            uint wt_grad_offset,
            uint bias_grad_offset,
            uint warp_id,
            bool accumulate_bias_grad = true
        )
        {
            var grad_w = outer_product_reduce(ip, grad);

            if (accumulate_bias_grad)
            {
                reduce_bias(grad, bias_grad, accumulate, bias_grad_offset, warp_id);
            }
            reduce_weights(grad_w, wts_grad, accumulate, wt_grad_offset, warp_id);

            HMatrixB<Z0, Z1> w;
            w.load_mma(wts_buff, wt_offset);
            var w_t = change_major_axis(transpose(w));
            var grad_ip = mul(grad, w_t);
            return grad_ip;
        }

        static uint weight_offset() { return Z0 * Z1 / half::NUM_PACKED; }

        static uint bias_offset() { return Z1 / half::NUM_PACKED; }
    };

    struct HMLPGrad<
        let WARPS_PER_WG : uint,
        let HIDDEN_LAYERS : uint,
        let Z_INPUT : uint,
        let Z_HIDDEN : uint,
        let Z_OUTPUT : uint,
        HIDDEN_ACT : IHActFnBackward & IHActFn,
        OUTPUT_ACT : IHActFnBackward & IHActFn>
    {
        static const uint MAX_WEIGHT = Z_INPUT > Z_HIDDEN? Z_INPUT* Z_HIDDEN : Z_OUTPUT > Z_HIDDEN? Z_OUTPUT* Z_HIDDEN : Z_HIDDEN* Z_HIDDEN;
        static const uint WEIGHT_SIZE = MAX_WEIGHT / half::NUM_PACKED;

        uint warp_id;

        __init(uint warp_id = 0) { this.warp_id = warp_id; }

        uint wt_offset;
        uint bias_offset;

        HLinearGrad<Z_INPUT, Z_HIDDEN, WARPS_PER_WG, WEIGHT_SIZE> ip_layer;
        HLinearGrad<Z_HIDDEN, Z_HIDDEN, WARPS_PER_WG, WEIGHT_SIZE> hidden_layer;
        HLinearGrad<Z_HIDDEN, Z_OUTPUT, WARPS_PER_WG, WEIGHT_SIZE> op_layer;

        HVector<Z_INPUT> ip_cached;
        HVector<Z_HIDDEN> hidden_cached[HIDDEN_LAYERS + 1];
        HVector<Z_OUTPUT> op_cached;

        uint last_layer_wt_offset() { return ip_layer.weight_offset() + hidden_layer.weight_offset() * HIDDEN_LAYERS; }

        uint last_layer_bias_offset() { return ip_layer.bias_offset() + hidden_layer.bias_offset() * HIDDEN_LAYERS; }

        [mutating]
        HVector<Z_OUTPUT> forward(
            HVector<Z_INPUT> ip,
            WeightBuff wts_buff,
            BiasBuff bias_buff,
            uint wt_base_offset = 0U,
            uint bias_base_offset = 0U,
        )
        {
            uint wt_offset = wt_base_offset;
            uint bias_offset = bias_base_offset;

            ip_cached = ip;
            var h = ip_layer.forward(ip, wts_buff, bias_buff, wt_offset, bias_offset);
            hidden_cached[0] = h;

            wt_offset += ip_layer.weight_offset();
            bias_offset += ip_layer.bias_offset();

            HIDDEN_ACT h_act;
            h = happly(h_act, h);

            [unroll]
            for (uint i = 0; i < HIDDEN_LAYERS; i++)
            {
                h = hidden_layer.forward(h, wts_buff, bias_buff, wt_offset, bias_offset);
                hidden_cached[i + 1] = h;

                wt_offset += hidden_layer.weight_offset();
                bias_offset += hidden_layer.bias_offset();
                h = happly(h_act, h);
            }

            OUTPUT_ACT op_act;
            var op = op_layer.forward(h, wts_buff, bias_buff, wt_offset, bias_offset);
            op_cached = op;
            op = happly(op_act, op);
            return op;
        }

        HVector<Z_INPUT> backward(
            HVector<Z_OUTPUT> grad,
            WeightBuff wts_buff,
            RWGradBuff wts_grad,
            RWGradBuff bias_grad,
            bool accumulate = false,
            uint wt_base_offset = 0U,
            uint bias_base_offset = 0U,
            uint wt_grad_base_offset = 0U,
            uint bias_grad_base_offset = 0U
        )
        {
            uint wt_offset = last_layer_wt_offset() + wt_base_offset;
            uint bias_offset = last_layer_bias_offset() + bias_base_offset;
            uint wt_grad_offset = last_layer_wt_offset() + wt_grad_base_offset;
            uint bias_grad_offset = last_layer_bias_offset() + bias_grad_base_offset;

            OUTPUT_ACT op_act;
            HIDDEN_ACT h_act;

            var layer_op = happly(h_act, hidden_cached[HIDDEN_LAYERS]);
            var grad_op = happly2x(op_act, grad, op_cached);
            var grad_h = op_layer.backward(
                layer_op,
                grad_op,
                wts_buff,
                wts_grad,
                bias_grad,
                accumulate,
                wt_offset,
                bias_offset,
                wt_grad_offset,
                bias_grad_offset,
                warp_id
            );

            HVector<Z_HIDDEN> layer_h;

            [unroll]
            for (int i = int(HIDDEN_LAYERS) - 1; i >= 0; i--)
            {
                wt_offset -= hidden_layer.weight_offset();
                bias_offset -= hidden_layer.bias_offset();
                wt_grad_offset -= hidden_layer.weight_offset();
                bias_grad_offset -= hidden_layer.bias_offset();

                grad_h = happly2x(h_act, grad_h, hidden_cached[i + 1]);
                layer_h = happly(h_act, hidden_cached[i]);
                grad_h = hidden_layer.backward(
                    layer_h,
                    grad_h,
                    wts_buff,
                    wts_grad,
                    bias_grad,
                    accumulate,
                    wt_offset,
                    bias_offset,
                    wt_grad_offset,
                    bias_grad_offset,
                    warp_id
                );
            }

            var grad_ip = happly2x(h_act, grad_h, hidden_cached[0]);
            var grad_res = ip_layer.backward(
                ip_cached,
                grad_ip,
                wts_buff,
                wts_grad,
                bias_grad,
                accumulate,
                wt_base_offset,
                bias_base_offset,
                wt_grad_base_offset,
                bias_grad_base_offset,
                warp_id
            );
            return grad_res;
        }
    };
};
