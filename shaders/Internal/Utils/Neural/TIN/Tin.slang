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

import Utils.NVAPI;
import Internal.Utils.Neural.ActivationFunctions;
__exported import Internal.Utils.Neural.TIN.TinCommon;

__target_intrinsic(hlsl) uint2 __nvvm_mma_16x8x16(uint64_t mode, uint4 a, uint2 b, uint2 c);
__target_intrinsic(hlsl) uint __nvvm_transpose_8x8(uint a);

[ForceInline]
Ref<uint> tinSharedGrad<let SMEM_SIZE : uint>(uint index)
{
    static groupshared uint grads[SMEM_SIZE];
    return grads[index];
}

namespace Tin
{
    // The HMMA instruction and matrix layout is described here:
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16816-float
    static uint2 mma_16x8x16(uint4 A, uint2 B, uint2 C)
    {
        const uint64_t mode = 0;
        return __nvvm_mma_16x8x16(mode, A, B, C);
    };

    static uint hmma_transpose(uint x)
    {
        return __nvvm_transpose_8x8(x);
    };

    static uint atomic_add_half2(RWByteAddressBuffer uav, uint byteAddress, uint x)
    {
        return NvInterlockedAddFp16x2(uav, byteAddress, x);
    };

    // Shuffles dwords across the lanes in a warp
    static uint shuffle(uint val, uint srcLane)
    {
        int ret = NvShfl(asint(val), srcLane, WAVE_SIZE);
        return asuint(ret);
    }

    // Shuffles dwords across the lanes in a warp
    static uint shuffle_down(uint val, uint delta)
    {
        int ret = NvShflDown(asint(val), delta, WAVE_SIZE);
        return asuint(ret);
    }

    // MMAMatrix is a warp wide storage for an 16x8 dword matrix (row major) or a 8x16 dword matrix (col major).
    // For 16-bit datatypes MMAMatrix size is 16x16 i.e. 16x8*2 (row-major) and 8*2x16 (col major).
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16816-float
    // For 8-bit datatypes it is a 16x32 (row major) or 32x16 matrix (col major).
    // MMAMatrix operations are mapped to tensor core intrinsics

    struct MMAMat<T : IPackable, let G_ROW_REGS : uint = 2, let G_COL_REGS : uint = 2, let ROW_MAJOR : bool = true>
    {
        static const uint ROW_REGS = G_ROW_REGS;
        static const uint COL_REGS = G_COL_REGS;
        static const uint N_REGS = ROW_REGS * COL_REGS;

        static const uint ROWS_PACKED = ROW_REGS * MMAReg::ROWS_PACKED;
        static const uint COLS_PACKED = COL_REGS * MMAReg::COLS_PACKED;
        static const uint N_PACKED = ROWS_PACKED * COLS_PACKED;

        static const uint ROWS = ROW_REGS * MMAReg::ROWS;
        static const uint COLS = COL_REGS * MMAReg::COLS;

        static const uint N_INNER = ROW_MAJOR ? COLS : ROWS;
        static const uint N_OUTER = ROW_MAJOR ? ROWS : COLS;

        MMAMatStore regs;

        static uint num_regs() { return ROW_REGS * COL_REGS; }

        static uint reg_row(uint row) { return row / MMAReg::ROWS_PACKED; }

        static uint reg_col(uint col) { return col / MMAReg::COLS_PACKED; }

        static uint subreg_row(uint row) { return row % MMAReg::ROWS_PACKED; }

        static uint subreg_col(uint col) { return col % MMAReg::COLS_PACKED; }

        static uint row_from_subreg(uint subreg_row, uint reg_row) { return subreg_row + reg_row * MMAReg::ROWS_PACKED; }

        static uint col_from_subreg(uint subreg_col, uint reg_col) { return subreg_col + reg_col * MMAReg::COLS_PACKED; }

        static uint reg_row_col_to_lin(uint reg_row, uint reg_col) { return reg_col * ROW_REGS + reg_row; }
        static uint row_col_to_lin(uint row, uint col)
        {
            return reg_row_col_to_lin(reg_row(row), reg_col(col)) * MMAReg::N_PACKED +
                   MMAReg::row_col_to_lin(subreg_row(row), subreg_col(col));
        }

        static uint row_col_to_mma_addr(uint row, uint col)
        {
            uint linear = row_col_to_lin(row, col);
            uint aos = (linear % MMAReg::N_PACKED) * N_REGS + linear / MMAReg::N_PACKED;
            return aos;
        }

        [mutating]
        void clear()
        {
            [unroll]
            for (uint i = 0; i < num_regs(); i++)
            {
                regs[i] = 0;
            }
        }

        [mutating]
        void fill(T::Vector x)
        {
            [unroll]
            for (uint i = 0; i < num_regs(); i++)
            {
                regs[i] = T::Vector::pack(x);
            }
        }

        // MMAReg is a single warp-wide register. It stores an 8x4 dword matrix (row major) or a 4x8 dword matrix (col major).
        // For 16-bit datatypes, it has 8x8 elements i.e. 8x4*2 (row major) or a 4*2x8 (col major).
        // For 8-bit datatypes, it has 8x4*2 = 8x32 elements (row major) or 32x8 elements (col major).
        struct MMAReg
        {
            static const uint N_OUTER = 8U;
            static const uint N_INNER_PACKED = WAVE_SIZE / N_OUTER;

            static const uint ROWS_PACKED = ROW_MAJOR ? N_OUTER : N_INNER_PACKED;
            static const uint COLS_PACKED = ROW_MAJOR ? N_INNER_PACKED : N_OUTER;
            static const uint N_PACKED = ROWS_PACKED * COLS_PACKED;

            static const uint N_INNER = N_INNER_PACKED * T::NUM_PACKED;
            static const uint ROWS = ROW_MAJOR ? N_OUTER : N_INNER;
            static const uint COLS = ROW_MAJOR ? N_INNER : N_OUTER;

            static uint row_col_to_lin(uint row, uint col)
            {
                if (ROW_MAJOR)
                {
                    return row * COLS_PACKED + col;
                }
                else
                {
                    return col * ROWS_PACKED + row;
                }
            }

            static uint2 lin_to_row_col(uint idx)
            {
                uint2 rc;
                if (ROW_MAJOR)
                {
                    rc.x = idx / COLS_PACKED;
                    rc.y = idx % COLS_PACKED;
                }
                else
                {
                    rc.y = idx / ROWS_PACKED;
                    rc.x = idx % ROWS_PACKED;
                }
                return rc;
            }
        };
    };

    typealias HMMAMat2x2<let ROW_MAJOR : bool> = MMAMat<half, 2, 2, ROW_MAJOR>;
    typealias HMMAMat2x2A = HMMAMat2x2<true>;
    typealias HMMAMat2x2B = HMMAMat2x2<false>;

    static HMMAMat2x2<ROW_MAJOR> operator +<let ROW_MAJOR : bool>(HMMAMat2x2<ROW_MAJOR> x, HMMAMat2x2<ROW_MAJOR> y)
    {
        HMMAMat2x2<ROW_MAJOR> r;
        [unroll]
        for (uint i = 0; i < HMMAMat2x2<ROW_MAJOR>::num_regs(); i++)
        {
            r.regs[i] = half2::pack(half2::unpack(x.regs[i]) + half2::unpack(y.regs[i]));
        }
        return r;
    }

    static void reduce_sum<let SIZE : uint>(HMMAMat2x2A x, uint dword_offset)
    {
        uint lane_id = WaveGetLaneIndex();
        uint2 subreg_rc = HMMAMat2x2A::MMAReg::lin_to_row_col(lane_id);

        [unroll]
        for (uint reg_col = 0; reg_col < HMMAMat2x2A::COL_REGS; reg_col++)
        {
            const uint ROW_REGS = HMMAMat2x2A::ROW_REGS;

            // sum row registers
            [unroll]
            for (uint j = 2; j <= ROW_REGS; j <<= 1)
            {
                [unroll]
                for (uint reg_row = 0; reg_row < ROW_REGS; reg_row += j)
                {
                    uint src_reg_idx = HMMAMat2x2A::reg_row_col_to_lin(reg_row + j / 2, reg_col);
                    uint dst_reg_idx = HMMAMat2x2A::reg_row_col_to_lin(reg_row, reg_col);
                    x.regs[dst_reg_idx] = half2::pack(half2::unpack(x.regs[dst_reg_idx]) + half2::unpack(x.regs[src_reg_idx]));
                }
            }

            uint reg_idx = HMMAMat2x2A::reg_row_col_to_lin(0, reg_col);

            const uint ROWS_PACKED = HMMAMat2x2A::MMAReg::ROWS_PACKED;
            const uint COLS_PACKED = HMMAMat2x2A::MMAReg::COLS_PACKED;

            // Binary reduce rows inside a register
            [unroll]
            for (uint j = 1; j < ROWS_PACKED; j <<= 1)
            {
                uint regs_shfl = shuffle_down(x.regs[reg_idx], COLS_PACKED * j);
                x.regs[reg_idx] = half2::pack(half2::unpack(x.regs[reg_idx]) + half2::unpack(regs_shfl));
            }

            // Write to memory
            uint col = reg_col * COLS_PACKED + subreg_rc.y;

            if (subreg_rc.x == 0)
            {
                tinSharedGrad<SIZE>(dword_offset + col) = x.regs[reg_idx];
            }
        }
    }

    // Low level matrix multiply-add for half-precision MMA matrices
    static HMMAMat2x2A mad(HMMAMat2x2A a, HMMAMat2x2B b, HMMAMat2x2A c)
    {
        HMMAMat2x2A r;

        r.regs.xy = mma_16x8x16(a.regs, b.regs.xy, c.regs.xy);
        r.regs.zw = mma_16x8x16(a.regs, b.regs.zw, c.regs.zw);

        return r;
    }

    static HMMAMat2x2A transpose(HMMAMat2x2B x)
    {
        HMMAMat2x2A r;

        [unroll]
        for (uint i = 0; i < HMMAMat2x2B::COL_REGS; i++)
        {
            [unroll]
            for (uint j = 0; j < HMMAMat2x2B::ROW_REGS; j++)
            {
                uint src_reg_idx = HMMAMat2x2B::reg_row_col_to_lin(j, i);
                uint dst_reg_idx = HMMAMat2x2B::reg_row_col_to_lin(i, j);
                r.regs[dst_reg_idx] = x.regs[src_reg_idx];
            }
        }
        return r;
    }

    // NOTE: Needs separate functions for HMMAMat2x2B and HMMAMat2x2A becase
    // older versions of slang crash on generic expression !ROW_MAJOR
    static HMMAMat2x2B transpose(HMMAMat2x2A x)
    {
        HMMAMat2x2B r;

        [unroll]
        for (uint i = 0; i < HMMAMat2x2A::COL_REGS; i++)
        {
            [unroll]
            for (uint j = 0; j < HMMAMat2x2A::ROW_REGS; j++)
            {
                uint src_reg_idx = HMMAMat2x2A::reg_row_col_to_lin(j, i);
                uint dst_reg_idx = HMMAMat2x2A::reg_row_col_to_lin(i, j);
                r.regs[dst_reg_idx] = x.regs[src_reg_idx];
            }
        }
        return r;
    }

    static HMMAMat2x2A change_major_axis(HMMAMat2x2B x)
    {
        HMMAMat2x2A r;

        [unroll]
        for (uint i = 0; i < HMMAMat2x2B::COL_REGS; i++)
        {
            [unroll]
            for (uint j = 0; j < HMMAMat2x2B::ROW_REGS; j++)
            {
                uint idx = HMMAMat2x2B::reg_row_col_to_lin(j, i);
                r.regs[idx] = hmma_transpose(x.regs[idx]);
            }
        }
        return r;
    }

    // NOTE: Needs separate functions for HMMAMat2x2B and HMMAMat2x2A becase
    // older versions of slang crash on generic expression !ROW_MAJOR
    static HMMAMat2x2B change_major_axis(HMMAMat2x2A x)
    {
        HMMAMat2x2B r;

        [unroll]
        for (uint i = 0; i < HMMAMat2x2A::COL_REGS; i++)
        {
            [unroll]
            for (uint j = 0; j < HMMAMat2x2A::ROW_REGS; j++)
            {
                uint idx = HMMAMat2x2A::reg_row_col_to_lin(j, i);
                r.regs[idx] = hmma_transpose(x.regs[idx]);
            }
        }
        return r;
    }

    // This is an arbitrary sized matrix that tiles hardware MMA matrices, each of which comprises 4 registers.
    struct Matrix<T : IPackable, let ROWS : uint, let COLS : uint, let ROW_MAJOR : bool = true>
    {
        typealias MMAMat2x2 = MMAMat<T, 2, 2, ROW_MAJOR>;

        static const uint MMA_ROWS = ROWS / MMAMat2x2::ROWS;
        static const uint MMA_COLS = COLS / MMAMat2x2::COLS;
        static const uint NUM_MMA = MMA_ROWS * MMA_COLS;

        static const uint ROWS_PACKED = MMAMat2x2::ROWS_PACKED * MMA_ROWS;
        static const uint COLS_PACKED = MMAMat2x2::COLS_PACKED * MMA_COLS;

        static const uint N_INNER = ROW_MAJOR ? COLS : ROWS;
        static const uint N_OUTER = ROW_MAJOR ? ROWS : COLS;

        MMAMat2x2 mma_mat[MMA_COLS][MMA_ROWS];

        static uint mma_load_offset() { return NUM_MMA * WAVE_SIZE; }

        static uint num_regs() { return NUM_MMA * MMAMat2x2::num_regs(); }

        static uint mma_mat_row(uint row) { return row / MMAMat2x2::ROWS_PACKED; }

        static uint mma_mat_col(uint col) { return col / MMAMat2x2::COLS_PACKED; }

        static uint submma_mat_row(uint row) { return row % MMAMat2x2::ROWS_PACKED; }

        static uint submma_mat_col(uint col) { return col % MMAMat2x2::COLS_PACKED; }

        static uint row_from_mat(uint submat_row, uint mat_row) { return submat_row + mat_row * MMAMat2x2::ROWS_PACKED; }

        static uint col_from_mat(uint submat_col, uint mat_col) { return submat_col + mat_col * MMAMat2x2::COLS_PACKED; }

        static uint2 mma_lin_to_row_col(uint mma_idx)
        {
            uint2 ret;
            ret.x = mma_idx % MMA_ROWS;
            ret.y = mma_idx / MMA_ROWS;
            return ret;
        }

        static uint mma_row_col_to_lin(uint mma_row, uint mma_col) { return mma_col * MMA_ROWS + mma_row; }
        static uint rc_to_addr_mma(uint row, uint col)
        {
            uint mma_row = mma_mat_row(row);
            uint mma_col = mma_mat_col(col);
            uint submma_row = submma_mat_row(row);
            uint submma_col = submma_mat_col(col);
            return MMAMat2x2::row_col_to_mma_addr(submma_row, submma_col) + mma_row_col_to_lin(mma_row, mma_col) * MMAMat2x2::N_PACKED;
        }

        [mutating]
        void clear()
        {
            [unroll]
            for (uint c = 0; c < MMA_COLS; c++)
            {
                [unroll]
                for (uint r = 0; r < MMA_ROWS; r++)
                {
                    mma_mat[c][r].clear();
                }
            }
        }

        [mutating]
        void fill(T::Vector x)
        {
            [unroll]
            for (uint c = 0; c < MMA_COLS; c++)
            {
                [unroll]
                for (uint r = 0; r < MMA_ROWS; r++)
                {
                    mma_mat[c][r].fill(x);
                }
            }
        }

#define OPTIMIZED_SHUFFLE

        [mutating]
        void from_array<TArr : IPackedArray>(TArr ip_array)
        {
            uint lane_id = WaveGetLaneIndex();

#ifdef OPTIMIZED_SHUFFLE
            const uint REG_COLS_PACKED = MMAMat2x2::MMAReg::COLS_PACKED;
            const uint REG_ROWS_PACKED = MMAMat2x2::MMAReg::ROWS_PACKED;

            [unroll]
            for (uint col_packed = 0; col_packed < max(TArr::SIZE_PACKED, MMAMat2x2::COLS_PACKED); col_packed += REG_COLS_PACKED)
            {
                uint regsi[MMAMat2x2::MMAReg::COLS_PACKED];

                [unroll]
                for (uint i = 0; i < REG_COLS_PACKED; i++)
                {
                    if (col_packed + i < TArr::SIZE_PACKED)
                        regsi[i] = ip_array.get_packed_item(col_packed + i);
                    else
                        regsi[i] = 0;

                    [unroll]
                    for (uint j = 1; j < REG_COLS_PACKED; j++)
                    {
                        uint val_packed = 0;
                        if (col_packed + (i + j) % REG_COLS_PACKED < TArr::SIZE_PACKED)
                            val_packed = ip_array.get_packed_item(col_packed + (i + j) % REG_COLS_PACKED);

                        if (lane_id / REG_ROWS_PACKED == j)
                        {
                            regsi[i] = val_packed;
                        }
                    }
                }

                [unroll]
                for (uint i = 0; i < REG_COLS_PACKED; i++)
                {
                    uint shfl_idx = ((lane_id + (REG_COLS_PACKED - i)) % REG_COLS_PACKED) * REG_ROWS_PACKED + lane_id / REG_COLS_PACKED;
                    regsi[i] = shuffle(regsi[i], shfl_idx);
                }

                uint mma_col = mma_mat_col(col_packed);
                uint submma_col = submma_mat_col(col_packed);
                uint r_col = MMAMat2x2::reg_col(submma_col);

                [unroll]
                for (uint i = 0; i < REG_COLS_PACKED; i++)
                {
                    uint r_row = i % MMAMat2x2::ROW_REGS;
                    uint mma_row = i / MMAMat2x2::ROW_REGS;

                    uint reg_lin = MMAMat2x2::reg_row_col_to_lin(r_row, r_col);

                    mma_mat[mma_col][mma_row].regs[reg_lin] = regsi[0];

                    [unroll]
                    for (uint j = 1; j < REG_COLS_PACKED; j++)
                    {
                        if (lane_id % REG_COLS_PACKED == (i + j) % REG_COLS_PACKED)
                        {
                            mma_mat[mma_col][mma_row].regs[reg_lin] = regsi[j];
                        }
                    }
                }
            }

#else
            uint2 subreg_rc = MMAMat2x2::MMAReg::lin_to_row_col(lane_id);
            uint mma_rows = mma_mat_row(WAVE_SIZE);

            [unroll]
            for (uint col_packed = 0; col_packed < TArr::SIZE_PACKED; col_packed++)
            {
                uint val_packed = ip_array.get_packed_item(col_packed);

                uint mma_col = mma_mat_col(col_packed);
                uint submma_col = submma_mat_col(col_packed);

                uint r_col = MMAMat2x2::reg_col(submma_col);
                uint s_col = MMAMat2x2::subreg_col(submma_col);

                [unroll]
                for (uint mma_row = 0; mma_row < mma_rows; mma_row++)
                {
                    [unroll]
                    for (uint r_row = 0; r_row < MMAMat2x2::ROW_REGS; r_row++)
                    {
                        // select a reg
                        uint reg_lin = MMAMat2x2::reg_row_col_to_lin(r_row, r_col);

                        uint32_t target_row = r_row * MMAMat2x2::MMAReg::ROWS_PACKED + mma_row * MMAMat2x2::ROWS_PACKED + subreg_rc.x;

                        // Shuffle input to align with register
                        uint val_shfl = shuffle(val_packed, target_row);

                        if (s_col == subreg_rc.y)
                        {
                            mma_mat[mma_col][mma_row].regs[reg_lin] = val_shfl;
                        }
                    }
                }
            }
#endif
        }

        // Elements of a matrix are always accesed and indexed as packed 32-bit values i.e. half2 for a matrix of type T=half
        // TODO: this should only be supproted for row major matrices
        void to_array<TArr : IPackedArray>(inout TArr op_array)
        {
            uint lane_id = WaveGetLaneIndex();

#ifdef OPTIMIZED_SHUFFLE

            const uint REG_COLS_PACKED = MMAMat2x2::MMAReg::COLS_PACKED;
            const uint REG_ROWS_PACKED = MMAMat2x2::MMAReg::ROWS_PACKED;

            [unroll]
            for (uint col_packed = 0; col_packed < TArr::SIZE_PACKED; col_packed += REG_COLS_PACKED)
            {
                uint regsi[MMAMat2x2::MMAReg::COLS_PACKED];

                uint mma_col = mma_mat_col(col_packed);
                uint submma_col = submma_mat_col(col_packed);
                uint r_col = MMAMat2x2::reg_col(submma_col);
                uint reg_lin = MMAMat2x2::reg_row_col_to_lin(0, r_col);

                [unroll]
                for (uint i = 0; i < REG_COLS_PACKED; i++)
                {
                    regsi[i] = mma_mat[mma_col][0].regs[reg_lin];

                    [unroll]
                    for (uint j = 1; j < REG_COLS_PACKED; j++)
                    {
                        uint r_row = j % MMAMat2x2::ROW_REGS;
                        uint mma_row = j / MMAMat2x2::ROW_REGS;
                        uint reg_lin = MMAMat2x2::reg_row_col_to_lin(r_row, r_col);

                        if (lane_id % REG_COLS_PACKED == (i + j) % REG_COLS_PACKED)
                        {
                            regsi[i] = mma_mat[mma_col][mma_row].regs[reg_lin];
                        }
                    }
                }

                [unroll]
                for (uint i = 0; i < REG_COLS_PACKED; i++)
                {
                    uint shfl_idx = (lane_id % REG_ROWS_PACKED) * REG_COLS_PACKED + (lane_id / REG_ROWS_PACKED + i) % REG_COLS_PACKED;
                    regsi[i] = shuffle(regsi[i], shfl_idx);
                }

                [unroll]
                for (uint i = 0; i < REG_COLS_PACKED; i++)
                {
                    uint op = regsi[0];

                    [unroll]
                    for (uint j = 1; j < REG_COLS_PACKED; j++)
                    {
                        if (lane_id / REG_ROWS_PACKED == (REG_COLS_PACKED - j + i) % REG_COLS_PACKED)
                        {
                            op = regsi[j];
                        }
                    }

                    if (col_packed + i < TArr::SIZE_PACKED)
                        op_array.set_packed_item(col_packed + i, op);
                }
            }

#else
            uint mma_rows = mma_mat_row(WAVE_SIZE);
            uint mma_row = mma_mat_row(lane_id);
            uint submma_row = submma_mat_row(lane_id);

            uint r_row = MMAMat2x2::reg_row(submma_row);
            uint s_row = MMAMat2x2::subreg_row(submma_row);

            [unroll]
            for (uint col_packed = 0; col_packed < TArr::SIZE_PACKED; col_packed++)
            {
                uint mma_col = mma_mat_col(col_packed);
                uint submma_col = submma_mat_col(col_packed);

                uint r_col = MMAMat2x2::reg_col(submma_col);
                uint s_col = MMAMat2x2::subreg_col(submma_col);

                uint dest = 0;

                [unroll]
                for (uint r = 0; r < mma_rows; r++)
                {
                    // Select a matrix
                    MMAMat2x2 mat = mma_mat[mma_col][r];

                    // compute source index inside register
                    uint subreg_lin = MMAMat2x2::MMAReg::row_col_to_lin(s_row, s_col);

                    [unroll]
                    for (uint i = 0; i < MMAMat2x2::ROW_REGS; i++)
                    {
                        // select a reg
                        uint reg_lin = MMAMat2x2::reg_row_col_to_lin(i, r_col);
                        uint reg = mat.regs[reg_lin];

                        // read register lanes
                        uint reg_shfl = shuffle(reg, subreg_lin);

                        if (mma_row == r && r_row == i)
                        {
                            dest = reg_shfl;
                        }
                    }
                }

                op_array.set_packed_item(col_packed, dest);
            }
#endif
        }

        [mutating]
        void merge(Matrix<T, ROWS, COLS, ROW_MAJOR> input, uint mask)
        {
            uint lane_id = WaveGetLaneIndex();
            uint2 subreg_rc = MMAMat2x2::MMAReg::lin_to_row_col(lane_id);

            [unroll]
            for (uint mma_row = 0; mma_row < MMA_ROWS; mma_row++)
            {
                [unroll]
                for (uint mma_col = 0; mma_col < MMA_COLS; mma_col++)
                {
                    [unroll]
                    for (uint reg_row = 0; reg_row < MMAMat2x2::ROW_REGS; reg_row++)
                    {
                        [unroll]
                        for (uint reg_col = 0; reg_col < MMAMat2x2::COL_REGS; reg_col++)
                        {
                            uint row = reg_row * MMAMat2x2::MMAReg::ROWS_PACKED + mma_row * MMAMat2x2::ROWS_PACKED + subreg_rc.x;
                            uint lin_reg_idx = MMAMat2x2::reg_row_col_to_lin(reg_row, reg_col);

                            if ((mask & (1 << row)) > 0)
                            {
                                mma_mat[mma_col][mma_row].regs[lin_reg_idx] = input.mma_mat[mma_col][mma_row].regs[lin_reg_idx];
                            }
                        }
                    }
                }
            }
        }

        static uint rc_to_addr_packed(uint row, uint col, uint inner_sub_stride, uint inner_stride)
        {
            uint sub_stride_packed = inner_sub_stride / T::NUM_PACKED;
            uint inner_stride_packed = inner_stride / T::NUM_PACKED;

            uint ck = col % sub_stride_packed;
            uint c1 = col / sub_stride_packed;
            uint rk = row % sub_stride_packed;
            uint r1 = row / sub_stride_packed;

            if (ROW_MAJOR)
            {
                return c1 * inner_stride_packed + row * sub_stride_packed + ck;
            }
            else
            {
                return r1 * inner_stride_packed + col * sub_stride_packed + rk;
            }
        }

        // Load from uint buffer
        //  Element addresses for a row-major matrix
        //  SS: Substride
        //  IS: Inner stride
        //  |0  , 1    , 2    , ... , SS-1      | IS    , 1 + IS    , 2+IS    , ... , SS-1+IS    | 2IS,
        //  |SS , 1+SS , 2+SS , ... , SS-1 +  SS| IS+SS , 1 + IS+SS , 2+IS+SS , ... , IS-1+IS+SS | 2IS+SS,
        //  |2SS, 1+2SS, 2+2SS, ... , SS-1 + 2SS| IS+2SS, 1 + IS+2SS, 2+IS+2SS, ... , IS-1+IS+2SS| 2IS+2SS,

        [mutating]
        void load(StructuredBuffer<uint> buffer_in, uint dword_offset, uint inner_sub_stride = N_INNER, uint inner_stride = 0)
        {
            uint lane_id = WaveGetLaneIndex();
            uint2 subreg_rc = MMAMat2x2::MMAReg::lin_to_row_col(lane_id);

            [unroll]
            for (uint mma_row = 0; mma_row < MMA_ROWS; mma_row++)
            {
                [unroll]
                for (uint mma_col = 0; mma_col < MMA_COLS; mma_col++)
                {
                    [unroll]
                    for (uint reg_row = 0; reg_row < MMAMat2x2::ROW_REGS; reg_row++)
                    {
                        [unroll]
                        for (uint reg_col = 0; reg_col < MMAMat2x2::COL_REGS; reg_col++)
                        {
                            uint row = reg_row * MMAMat2x2::MMAReg::ROWS_PACKED + mma_row * MMAMat2x2::ROWS_PACKED + subreg_rc.x;

                            uint col = reg_col * MMAMat2x2::MMAReg::COLS_PACKED + mma_col * MMAMat2x2::COLS_PACKED + subreg_rc.y;

                            uint lin_reg_idx = MMAMat2x2::reg_row_col_to_lin(reg_row, reg_col);
                            uint src_idx = dword_offset + rc_to_addr_packed(row, col, inner_sub_stride, inner_stride);

                            mma_mat[mma_col][mma_row].regs[lin_reg_idx] = buffer_in[src_idx];
                        }
                    }
                }
            }
        }

        // Broadcast the elements across rows
        [mutating]
        void load_uniform(StructuredBuffer<uint> buffer_in, uint dword_offset)
        {
            uint lane_id = WaveGetLaneIndex();
            uint2 subreg_rc = MMAMat2x2::MMAReg::lin_to_row_col(lane_id);

            [unroll]
            for (uint mma_col = 0; mma_col < MMA_COLS; mma_col++)
            {
                [unroll]
                for (uint reg_col = 0; reg_col < MMAMat2x2::COL_REGS; reg_col++)
                {
                    uint col = reg_col * MMAMat2x2::MMAReg::COLS_PACKED + mma_col * MMAMat2x2::COLS_PACKED + subreg_rc.y;

                    uint lin_reg_idx = MMAMat2x2::reg_row_col_to_lin(0, reg_col);
                    mma_mat[mma_col][0].regs[lin_reg_idx] = buffer_in[dword_offset + col];
                }
            }

            [unroll]
            for (uint mma_row = 0; mma_row < MMA_ROWS; mma_row++)
            {
                [unroll]
                for (uint mma_col = 0; mma_col < MMA_COLS; mma_col++)
                {
                    [unroll]
                    for (uint reg_row = 0; reg_row < MMAMat2x2::ROW_REGS; reg_row++)
                    {
                        [unroll]
                        for (uint reg_col = 0; reg_col < MMAMat2x2::COL_REGS; reg_col++)
                        {
                            uint src_reg_idx = MMAMat2x2::reg_row_col_to_lin(0, reg_col);
                            uint dst_reg_idx = MMAMat2x2::reg_row_col_to_lin(reg_row, reg_col);

                            mma_mat[mma_col][mma_row].regs[dst_reg_idx] = mma_mat[mma_col][0].regs[src_reg_idx];
                        }
                    }
                }
            }
        }

        // Store to uint buffer
        //  Element addresses for a row-major matrix
        //  SS: Substride
        //  IS: Inner stride
        //  |0  , 1    , 2    , ... , SS-1      | IS    , 1 + IS    , 2+IS    , ... , SS-1+IS    | 2IS,
        //  |SS , 1+SS , 2+SS , ... , SS-1 +  SS| IS+SS , 1 + IS+SS , 2+IS+SS , ... , IS-1+IS+SS | 2IS+SS,
        //  |2SS, 1+2SS, 2+2SS, ... , SS-1 + 2SS| IS+2SS, 1 + IS+2SS, 2+IS+2SS, ... , IS-1+IS+2SS| 2IS+2SS,

        void store(RWStructuredBuffer<uint> buffer_out, uint dword_offset, uint inner_sub_stride = N_INNER, uint inner_stride = 0)
        {
            uint lane_id = WaveGetLaneIndex();
            uint2 subreg_rc = MMAMat2x2::MMAReg::lin_to_row_col(lane_id);

            [unroll]
            for (uint mma_row = 0; mma_row < MMA_ROWS; mma_row++)
            {
                [unroll]
                for (uint mma_col = 0; mma_col < MMA_COLS; mma_col++)
                {
                    [unroll]
                    for (uint reg_row = 0; reg_row < MMAMat2x2::ROW_REGS; reg_row++)
                    {
                        [unroll]
                        for (uint reg_col = 0; reg_col < MMAMat2x2::COL_REGS; reg_col++)
                        {
                            uint row = reg_row * MMAMat2x2::MMAReg::ROWS_PACKED + mma_row * MMAMat2x2::ROWS_PACKED + subreg_rc.x;

                            uint col = reg_col * MMAMat2x2::MMAReg::COLS_PACKED + mma_col * MMAMat2x2::COLS_PACKED + subreg_rc.y;

                            uint lin_reg_idx = MMAMat2x2::reg_row_col_to_lin(reg_row, reg_col);
                            uint dst_idx = dword_offset + rc_to_addr_packed(row, col, inner_sub_stride, inner_stride);

                            buffer_out[dst_idx] = mma_mat[mma_col][mma_row].regs[lin_reg_idx];
                        }
                    }
                }
            }
        }

        void store_uniform(RWStructuredBuffer<uint> buffer_out, uint dword_offset)
        {
            uint lane_id = WaveGetLaneIndex();
            uint2 subreg_rc = MMAMat2x2::MMAReg::lin_to_row_col(lane_id);

            [unroll]
            for (uint mma_col = 0; mma_col < MMA_COLS; mma_col++)
            {
                [unroll]
                for (uint reg_col = 0; reg_col < MMAMat2x2::COL_REGS; reg_col++)
                {
                    uint col = reg_col * MMAMat2x2::MMAReg::COLS_PACKED + mma_col * MMAMat2x2::COLS_PACKED + subreg_rc.y;

                    uint lin_reg_idx = MMAMat2x2::reg_row_col_to_lin(0, reg_col);
                    uint dst_idx = dword_offset + col;

                    if (subreg_rc.x == 0)
                    {
                        buffer_out[dst_idx] = mma_mat[mma_col][0].regs[lin_reg_idx];
                    }
                }
            }
        }

        // Load/Store for uint buffer with native mma substride. substride is 8 for 16-bit types.
        //  Element addresses for a row-major matrix
        //  SS: Substride
        //  OS: Outer stride
        //  IS: Inner stride
        //  |0  , 1    , 2    , ... , SS-1      | IS    , 1 + IS    , 2+IS    , ... , SS-1+IS    | 2IS,
        //  |OS , 1+OS , 2+OS , ... , SS-1 +  OS| IS+OS , 1 + IS+OS , 2+IS+OS , ... , IS-1+IS+OS | 2IS+OS,
        //  |2OS, 1+2OS, 2+2OS, ... , SS-1 + 2OS| IS+2OS, 1 + IS+2OS, 2+IS+2OS, ... , IS-1+IS+2OS| 2IS+2OS,

        [mutating]
        void load_mma_stride(StructuredBuffer<uint> buffer_in, uint dword_offset)
        {
            const uint REG_INNER = MMAMat2x2::MMAReg::N_INNER;
            load(buffer_in, dword_offset, REG_INNER, REG_INNER * N_OUTER);
        }

        void store_mma_stride(RWStructuredBuffer<uint> buffer_out, uint dword_offset)
        {
            const uint REG_INNER = MMAMat2x2::MMAReg::N_INNER;
            store(buffer_out, dword_offset, REG_INNER, REG_INNER * N_OUTER);
        }

        // Load from uint shared memory
        //  Element addresses for a row-major matrix
        //  SS: Substride
        //  IS: Inner stride
        //  |0  , 1    , 2    , ... , SS-1      | IS    , 1 + IS    , 2+IS    , ... , SS-1+IS    | 2IS,
        //  |SS , 1+SS , 2+SS , ... , SS-1 +  SS| IS+SS , 1 + IS+SS , 2+IS+SS , ... , IS-1+IS+SS | 2IS+SS,
        //  |2SS, 1+2SS, 2+2SS, ... , SS-1 + 2SS| IS+2SS, 1 + IS+2SS, 2+IS+2SS, ... , IS-1+IS+2SS| 2IS+2SS,
        [mutating]
        void load<let SIZE : uint>(uint dword_offset, uint inner_sub_stride = N_INNER, uint inner_stride = 0)
        {
            uint lane_id = WaveGetLaneIndex();
            uint2 subreg_rc = MMAMat2x2::MMAReg::lin_to_row_col(lane_id);

            [unroll]
            for (uint mma_row = 0; mma_row < MMA_ROWS; mma_row++)
            {
                [unroll]
                for (uint mma_col = 0; mma_col < MMA_COLS; mma_col++)
                {
                    [unroll]
                    for (uint reg_row = 0; reg_row < MMAMat2x2::ROW_REGS; reg_row++)
                    {
                        [unroll]
                        for (uint reg_col = 0; reg_col < MMAMat2x2::COL_REGS; reg_col++)
                        {
                            uint row = reg_row * MMAMat2x2::MMAReg::ROWS_PACKED + mma_row * MMAMat2x2::ROWS_PACKED + subreg_rc.x;

                            uint col = reg_col * MMAMat2x2::MMAReg::COLS_PACKED + mma_col * MMAMat2x2::COLS_PACKED + subreg_rc.y;

                            uint lin_reg_idx = MMAMat2x2::reg_row_col_to_lin(reg_row, reg_col);
                            uint src_idx = dword_offset + rc_to_addr_packed(row, col, inner_sub_stride, inner_stride);

                            mma_mat[mma_col][mma_row].regs[lin_reg_idx] = tinSharedGrad<SIZE>(src_idx);
                        }
                    }
                }
            }
        }

        // Broadcast the elements across rows
        [mutating, ForceInline]
        void load_uniform<let SIZE : uint>(uint dword_offset)
        {
            uint lane_id = WaveGetLaneIndex();
            uint2 subreg_rc = MMAMat2x2::MMAReg::lin_to_row_col(lane_id);

            [unroll]
            for (uint mma_col = 0; mma_col < MMA_COLS; mma_col++)
            {
                [unroll]
                for (uint reg_col = 0; reg_col < MMAMat2x2::COL_REGS; reg_col++)
                {
                    uint col = reg_col * MMAMat2x2::MMAReg::COLS_PACKED + mma_col * MMAMat2x2::COLS_PACKED + subreg_rc.y;

                    uint lin_reg_idx = MMAMat2x2::reg_row_col_to_lin(0, reg_col);
                    mma_mat[mma_col][0].regs[lin_reg_idx] = tinSharedGrad<SIZE>(dword_offset + col);
                }
            }

            [unroll]
            for (uint mma_row = 0; mma_row < MMA_ROWS; mma_row++)
            {
                [unroll]
                for (uint mma_col = 0; mma_col < MMA_COLS; mma_col++)
                {
                    [unroll]
                    for (uint reg_row = 0; reg_row < MMAMat2x2::ROW_REGS; reg_row++)
                    {
                        [unroll]
                        for (uint reg_col = 0; reg_col < MMAMat2x2::COL_REGS; reg_col++)
                        {
                            uint src_reg_idx = MMAMat2x2::reg_row_col_to_lin(0, reg_col);
                            uint dst_reg_idx = MMAMat2x2::reg_row_col_to_lin(reg_row, reg_col);

                            mma_mat[mma_col][mma_row].regs[dst_reg_idx] = mma_mat[mma_col][0].regs[src_reg_idx];
                        }
                    }
                }
            }
        }

        // Store to uint shared memory
        //  Element addresses for a row-major matrix
        //  SS: Substride
        //  IS: Inner stride
        //  |0  , 1    , 2    , ... , SS-1      | IS    , 1 + IS    , 2+IS    , ... , SS-1+IS    | 2IS,
        //  |SS , 1+SS , 2+SS , ... , SS-1 +  SS| IS+SS , 1 + IS+SS , 2+IS+SS , ... , IS-1+IS+SS | 2IS+SS,
        //  |2SS, 1+2SS, 2+2SS, ... , SS-1 + 2SS| IS+2SS, 1 + IS+2SS, 2+IS+2SS, ... , IS-1+IS+2SS| 2IS+2SS
        void store<let SIZE : uint>(uint dword_offset, uint inner_sub_stride = N_INNER, uint inner_stride = 0)
        {
            uint lane_id = WaveGetLaneIndex();
            uint2 subreg_rc = MMAMat2x2::MMAReg::lin_to_row_col(lane_id);

            [unroll]
            for (uint mma_row = 0; mma_row < MMA_ROWS; mma_row++)
            {
                [unroll]
                for (uint mma_col = 0; mma_col < MMA_COLS; mma_col++)
                {
                    [unroll]
                    for (uint reg_row = 0; reg_row < MMAMat2x2::ROW_REGS; reg_row++)
                    {
                        [unroll]
                        for (uint reg_col = 0; reg_col < MMAMat2x2::COL_REGS; reg_col++)
                        {
                            uint row = reg_row * MMAMat2x2::MMAReg::ROWS_PACKED + mma_row * MMAMat2x2::ROWS_PACKED + subreg_rc.x;

                            uint col = reg_col * MMAMat2x2::MMAReg::COLS_PACKED + mma_col * MMAMat2x2::COLS_PACKED + subreg_rc.y;

                            uint lin_reg_idx = MMAMat2x2::reg_row_col_to_lin(reg_row, reg_col);
                            uint dst_idx = dword_offset + rc_to_addr_packed(row, col, inner_sub_stride, inner_stride);

                            tinSharedGrad<SIZE>(dst_idx) = mma_mat[mma_col][mma_row].regs[lin_reg_idx];
                        }
                    }
                }
            }
        }

        void store_uniform<let SIZE : uint>(uint dword_offset)
        {
            uint lane_id = WaveGetLaneIndex();
            uint2 subreg_rc = MMAMat2x2::MMAReg::lin_to_row_col(lane_id);

            [unroll]
            for (uint mma_col = 0; mma_col < MMA_COLS; mma_col++)
            {
                [unroll]
                for (uint reg_col = 0; reg_col < MMAMat2x2::COL_REGS; reg_col++)
                {
                    uint col = reg_col * MMAMat2x2::MMAReg::COLS_PACKED + mma_col * MMAMat2x2::COLS_PACKED + subreg_rc.y;

                    uint lin_reg_idx = MMAMat2x2::reg_row_col_to_lin(0, reg_col);
                    uint dst_idx = dword_offset + col;

                    if (subreg_rc.x == 0)
                    {
                        tinSharedGrad<SIZE>(dst_idx) = mma_mat[mma_col][0].regs[lin_reg_idx];
                    }
                }
            }
        }

        // Load from uint4 buffer using a native layout for MMA Matrices
        [mutating]
        void load_mma(StructuredBuffer<MMAMatStore> buffer_in, uint dword_offset)
        {
            uint mma_offset = dword_offset / 4;
            uint lane_id = WaveGetLaneIndex();

            [unroll]
            for (uint c = 0; c < MMA_COLS; c++)
            {
                [unroll]
                for (uint r = 0; r < MMA_ROWS; r++)
                {
                    uint offset = mma_offset + (c * MMA_ROWS + r) * WAVE_SIZE + lane_id;
                    mma_mat[c][r].regs = buffer_in[offset];
                }
            }
        }

        // Store to uint4 buffer using a native layout for MMA Matrices
        void store_mma(RWStructuredBuffer<MMAMatStore> buffer_out, uint dword_offset)
        {
            uint mma_offset = dword_offset / 4;
            uint lane_id = WaveGetLaneIndex();

            [unroll]
            for (uint c = 0; c < MMA_COLS; c++)
            {
                [unroll]
                for (uint r = 0; r < MMA_ROWS; r++)
                {
                    uint offset = mma_offset + (c * MMA_ROWS + r) * WAVE_SIZE + lane_id;
                    buffer_out[offset] = mma_mat[c][r].regs;
                }
            }
        }

        // Load from uint shared memory using a native layout for MMA Matrices
        [mutating]
        void load_mma<let SIZE : uint>(uint dword_offset)
        {
            uint lane_id = WaveGetLaneIndex();
            const uint NUM_REGS = MMAMat2x2::num_regs();
            [unroll]
            for (uint c = 0; c < MMA_COLS; c++)
            {
                [unroll]
                for (uint r = 0; r < MMA_ROWS; r++)
                {
                    [unroll]
                    for (uint i = 0; i < NUM_REGS; i++)
                    {
                        uint offset = dword_offset + (c * MMA_ROWS + r) * NUM_REGS * WAVE_SIZE + i * WAVE_SIZE + lane_id;
                        mma_mat[c][r].regs[i] = tinSharedGrad<SIZE>(offset);
                    }
                }
            }
        }

        // Load to uint shared memory using a native layout for MMA Matrices
        void store_mma<let SIZE : uint>(uint dword_offset)
        {
            uint lane_id = WaveGetLaneIndex();
            const uint NUM_REGS = MMAMat2x2::num_regs();
            [unroll]
            for (uint c = 0; c < MMA_COLS; c++)
            {
                [unroll]
                for (uint r = 0; r < MMA_ROWS; r++)
                {
                    [unroll]
                    for (uint i = 0; i < NUM_REGS; i++)
                    {
                        uint offset = dword_offset + (c * MMA_ROWS + r) * NUM_REGS * WAVE_SIZE + i * WAVE_SIZE + lane_id;
                        tinSharedGrad<SIZE>(offset) = mma_mat[c][r].regs[i];
                    }
                }
            }
        }

        [mutating]
        void set_reg_packed(uniform uint reg_index, T::Vector val)
        {
            uint val_packed = T::Vector::pack(val);
            uint mma_index = reg_index / MMAMat2x2::num_regs();
            uint mma_reg_index = reg_index % MMAMat2x2::num_regs();
            uint2 rc = mma_lin_to_row_col(mma_index);
            mma_mat[rc.y][rc.x].regs[mma_reg_index] = val_packed;
        }

        T::Vector get_reg_packed(uniform uint reg_index)
        {
            uint mma_index = reg_index / MMAMat2x2::num_regs();
            uint mma_reg_index = reg_index % MMAMat2x2::num_regs();
            uint2 rc = mma_lin_to_row_col(mma_index);
            return T::Vector::unpack(mma_mat[rc.y][rc.x].regs[mma_reg_index]);
        }
    };

    typealias HMatrix<let ROWS : uint, let COLS : uint, let ROW_MAJOR : bool> = Tin::Matrix<half, ROWS, COLS, ROW_MAJOR>;
    typealias HMatrixA<let ROWS : uint, let COLS : uint> = HMatrix<ROWS, COLS, true>;
    typealias HMatrixB<let ROWS : uint, let COLS : uint> = HMatrix<ROWS, COLS, false>;

    typealias Vector<T : IPackable, let COLS : uint> = Tin::Matrix<T, WAVE_SIZE, COLS, true>;
    typealias HVector<let COLS : uint> = Vector<half, COLS>;

    static void hload_and_accumulate_mma<let ROWS : uint, let COLS : uint, let SIZE : uint>(
        RWByteAddressBuffer buffer_out, uint src_dw_offset, uint dst_dw_offset, bool accum = false
    )
    {
        const uint MMA_ROWS = HMatrixB<ROWS, COLS>::MMA_ROWS;
        const uint MMA_COLS = HMatrixB<ROWS, COLS>::MMA_COLS;
        const uint NUM_REGS = HMMAMat2x2B::num_regs();

        uint lane_id = WaveGetLaneIndex();

        [unroll]
        for (uint mma_row = 0; mma_row < MMA_ROWS; mma_row++)
        {
            [unroll]
            for (uint mma_col = 0; mma_col < MMA_COLS; mma_col++)
            {
                [unroll]
                for (uint i = 0; i < NUM_REGS; i++)
                {
                    uint mma_idx = (mma_col * MMA_ROWS + mma_row) * NUM_REGS * WAVE_SIZE;
                    uint reg_idx = i * WAVE_SIZE + lane_id;

                    uint src_a = reg_idx / NUM_REGS;
                    uint src_b = reg_idx % NUM_REGS;

                    uint src_offset = src_dw_offset + mma_idx + src_b * WAVE_SIZE + src_a;
                    uint dst_offset = dst_dw_offset + mma_idx + reg_idx;

                    uint reg = tinSharedGrad<SIZE>(src_offset);

                    if (accum)
                        atomic_add_half2(buffer_out, dst_offset * 4, reg);
                    else
                        buffer_out.Store(dst_offset * 4, reg);
                }
            }
        }
    }

    static void hload_and_accumulate_uniform<let COLS : uint, let SIZE : uint>(
        RWByteAddressBuffer buffer_out, uint src_dw_offset, uint dst_dw_offset, bool accum = false
    )
    {
        uint lane_id = WaveGetLaneIndex();
        uint COLS_PACKED = COLS / half::NUM_PACKED;
        uint COLS_PER_THREAD = max(1, COLS_PACKED / WAVE_SIZE);

        [unroll]
        for (uint i = 0; i < COLS_PER_THREAD; i++)
        {
            uint col = lane_id + WAVE_SIZE * i;
            if (col < COLS_PACKED)
            {
                uint reg = tinSharedGrad<SIZE>(src_dw_offset + col);
                uint dst_offset = dst_dw_offset + +col;

                if (accum)
                    atomic_add_half2(buffer_out, dst_offset * 4, reg);
                else
                    buffer_out.Store(dst_offset * 4, reg);
            }
        }
    }

    // Reduce elements across warp
    static void reduce_sum<let COLS : uint, let SIZE : uint>(HVector<COLS> x, uint dword_offset = 0)
    {
        const uint MMA_ROWS = HVector<COLS>::MMA_ROWS;
        const uint MMA_COLS = HVector<COLS>::MMA_COLS;

        HVector<COLS> y = x;

        // reduce MMAs across rows
        [unroll]
        for (uint mma_col = 0; mma_col < MMA_COLS; mma_col++)
        {
            [unroll]
            for (uint j = 2; j <= MMA_ROWS; j <<= 1)
            {
                [unroll]
                for (uint mma_row = 0; mma_row < MMA_ROWS; mma_row += j)
                {
                    y.mma_mat[mma_col][mma_row] = y.mma_mat[mma_col][mma_row] + y.mma_mat[mma_col][mma_row + j / 2];
                }
            }
        }

        // reduce inside MMA
        [unroll]
        for (uint mma_col = 0; mma_col < MMA_COLS; mma_col++)
        {
            uint packed_col = mma_col * HMMAMat2x2A::COLS_PACKED;
            reduce_sum<SIZE>(y.mma_mat[mma_col][0], dword_offset + packed_col);
        }
    };

    static HMatrix<ROWS0, COLS0, ROW_MAJOR> operator +
        <let ROWS0 : uint, let COLS0 : uint, let ROW_MAJOR : bool>(HMatrix<ROWS0, COLS0, ROW_MAJOR> x, HMatrix<ROWS0, COLS0, ROW_MAJOR> y)
    {
        const uint MMA_ROWS = HMatrix<ROWS0, COLS0, ROW_MAJOR>::MMA_ROWS;
        const uint MMA_COLS = HMatrix<ROWS0, COLS0, ROW_MAJOR>::MMA_COLS;

        HMatrix<ROWS0, COLS0, ROW_MAJOR> r;

        [unroll]
        for (uint mma_row = 0; mma_row < MMA_ROWS; mma_row++)
        {
            [unroll]
            for (uint mma_col = 0; mma_col < MMA_COLS; mma_col++)
            {
                r.mma_mat[mma_col][mma_row] = x.mma_mat[mma_col][mma_row] + y.mma_mat[mma_col][mma_row];
            }
        }

        return r;
    }

    static HMatrixA<ROWS0, COLS1> mad<let ROWS0 : uint, let COLS0 : uint, let COLS1 : uint>(
        HMatrixA<ROWS0, COLS0> a, HMatrixB<COLS0, COLS1> b, HMatrixA<ROWS0, COLS1> c
    )
    {
        Matrix<half, ROWS0, COLS1, true> res = c;

        const uint MMA_ROWS = HMatrixA<ROWS0, COLS1>::MMA_ROWS;
        const uint MMA_COLS = HMatrixA<ROWS0, COLS1>::MMA_COLS;
        const uint K = HMatrixA<ROWS0, COLS0>::MMA_COLS;

        [unroll]
        for (uint mma_row = 0; mma_row < MMA_ROWS; mma_row++)
        {
            [unroll]
            for (uint mma_col = 0; mma_col < MMA_COLS; mma_col++)
            {
                [unroll]
                for (uint k = 0; k < K; k++)
                {
                    res.mma_mat[mma_col][mma_row] = mad(a.mma_mat[k][mma_row], b.mma_mat[mma_col][k], res.mma_mat[mma_col][mma_row]);
                }
            }
        }

        return res;
    };

    static HMatrixA<ROWS0, COLS1> mul<let ROWS0 : uint, let COLS0 : uint, let COLS1 : uint>(
        HMatrixA<ROWS0, COLS0> a, HMatrixB<COLS0, COLS1> b
    )
    {
        HMatrixA<ROWS0, COLS1> c;
        c.clear();
        return mad(a, b, c);
    }

    static HMatrixA<COLS, ROWS> transpose<let ROWS : uint, let COLS : uint>(HMatrixB<ROWS, COLS> x)
    {
        HMatrixA<COLS, ROWS> r;
        const uint MMA_COLS = HMatrixB<ROWS, COLS>::MMA_COLS;
        const uint MMA_ROWS = HMatrixB<ROWS, COLS>::MMA_ROWS;

        [unroll]
        for (uint i = 0; i < MMA_COLS; i++)
        {
            [unroll]
            for (uint j = 0; j < MMA_ROWS; j++)
            {
                r.mma_mat[j][i] = transpose(x.mma_mat[i][j]);
            }
        }

        return r;
    }

    // NOTE: Needs separate functions for HMMAMat2x2B and HMMAMat2x2A becase
    // older versions of slang crash on generic expression !ROW_MAJOR
    static HMatrixB<COLS, ROWS> transpose<let ROWS : uint, let COLS : uint>(HMatrixA<ROWS, COLS> x)
    {
        HMatrixB<COLS, ROWS> r;
        const uint MMA_COLS = HMatrixA<ROWS, COLS>::MMA_COLS;
        const uint MMA_ROWS = HMatrixA<ROWS, COLS>::MMA_ROWS;

        [unroll]
        for (uint i = 0; i < MMA_COLS; i++)
        {
            [unroll]
            for (uint j = 0; j < MMA_ROWS; j++)
            {
                r.mma_mat[j][i] = transpose(x.mma_mat[i][j]);
            }
        }

        return r;
    }

    static HMatrixA<ROWS, COLS> change_major_axis<let ROWS : uint, let COLS : uint>(HMatrixB<ROWS, COLS> x)
    {
        HMatrixA<ROWS, COLS> r;
        const uint MMA_COLS = HMatrixB<ROWS, COLS>::MMA_COLS;
        const uint MMA_ROWS = HMatrixB<ROWS, COLS>::MMA_ROWS;

        [unroll]
        for (uint i = 0; i < MMA_COLS; i++)
        {
            [unroll]
            for (uint j = 0; j < MMA_ROWS; j++)
            {
                r.mma_mat[i][j] = change_major_axis(x.mma_mat[i][j]);
            }
        }
        return r;
    };

    // NOTE: Needs separate functions for HMMAMat2x2B and HMMAMat2x2A becase
    // older versions of slang crash on generic expression !ROW_MAJOR
    static HMatrixB<ROWS, COLS> change_major_axis<let ROWS : uint, let COLS : uint>(HMatrixA<ROWS, COLS> x)
    {
        HMatrixB<ROWS, COLS> r;
        const uint MMA_COLS = HMatrixA<ROWS, COLS>::MMA_COLS;
        const uint MMA_ROWS = HMatrixA<ROWS, COLS>::MMA_ROWS;

        [unroll]
        for (uint i = 0; i < MMA_COLS; i++)
        {
            [unroll]
            for (uint j = 0; j < MMA_ROWS; j++)
            {
                r.mma_mat[i][j] = change_major_axis(x.mma_mat[i][j]);
            }
        }
        return r;
    };

    static HMatrixB<COLS0, COLS1> outer_product_reduce_acc<let COLS0 : uint, let COLS1 : uint>(
        HVector<COLS0> a, HVector<COLS1> b, HMatrixB<COLS0, COLS1> c
    )
    {
        HMatrixB<WAVE_SIZE, COLS0> a_t = change_major_axis(a);
        HMatrixA<COLS1, WAVE_SIZE> b_t = change_major_axis(transpose(b));
        HMatrixA<COLS1, COLS0> c_t = transpose(c);

        HMatrixA<COLS1, COLS0> w = mad(b_t, a_t, c_t);
        HMatrixB<COLS0, COLS1> w_t = transpose(w);
        return w_t;
    }

    static HMatrixB<COLS0, COLS1> outer_product_reduce<let COLS0 : uint, let COLS1 : uint>(HVector<COLS0> a, HVector<COLS1> b)
    {
        HMatrixB<WAVE_SIZE, COLS0> a_t = change_major_axis(a);
        HMatrixA<COLS1, WAVE_SIZE> b_t = change_major_axis(transpose(b));

        HMatrixA<COLS1, COLS0> w = mul(b_t, a_t);
        HMatrixB<COLS0, COLS1> w_t = transpose(w);
        return w_t;
    }

    static HMatrixA<ROWS, COLS> happly<Act : IHActFn, let ROWS : uint, let COLS : uint>(Act f, HMatrixA<ROWS, COLS> x)
    {
        HMatrixA<ROWS, COLS> out;

        [unroll]
        for (uint i = 0; i < x.num_regs(); i++)
        {
            half2 d = x.get_reg_packed(i);
            d.x = f.eval(d.x);
            d.y = f.eval(d.y);
            out.set_reg_packed(i, d);
        }

        return out;
    }

    static HMatrixA<ROWS, COLS> happly2x<Act : IHActFnBackward, let ROWS : uint, let COLS : uint>(
        Act f, HMatrixA<ROWS, COLS> ma, HMatrixA<ROWS, COLS> mb
    )
    {
        HMatrixA<ROWS, COLS> out;

        [unroll]
        for (uint i = 0; i < ma.num_regs(); i++)
        {
            half2 a = ma.get_reg_packed(i);
            half2 b = mb.get_reg_packed(i);
            half2 d;

            d.x = f.backward(a.x, b.x);
            d.y = f.backward(a.y, b.y);

            out.set_reg_packed(i, d);
        }

        return out;
    }

    static HVector<Z1> mad<let Z0 : uint, let Z1 : uint>(
        HVector<Z0> ip, HVector<Z1> bias, StructuredBuffer<Tin::MMAMatStore> wts_buff, uint wt_offset = 0U
    )
    {
        HMatrixB<Z0, Z1> wt;
        wt.load_mma(wts_buff, wt_offset);

        HVector<Z1> out;
        out = mad(ip, wt, bias);
        return out;
    }

    static HVector<Z1> mad<let Z0 : uint, let Z1 : uint>(
        HVector<Z0> ip, StructuredBuffer<Tin::MMAMatStore> wts_buff, uint wt_offset, StructuredBuffer<uint> bias_buff, uint bias_offset
    )
    {
        HMatrixB<Z0, Z1> wt;
        wt.load_mma(wts_buff, wt_offset);

        HVector<Z1> bias;
        bias.load_uniform(bias_buff, bias_offset);

        HVector<Z1> out;
        out = mad(ip, wt, bias);
        return out;
    }

    static HVector<Z1> mul<let Z0 : uint, let Z1 : uint>(HVector<Z0> ip, StructuredBuffer<Tin::MMAMatStore> wts_buff, uint wt_offset)
    {
        HMatrixB<Z0, Z1> wt;
        wt.load_mma(wts_buff, wt_offset);

        HVector<Z1> bias;
        bias.clear();

        HVector<Z1> out;
        out = mad(ip, wt, bias);
        return out;
    }
};
