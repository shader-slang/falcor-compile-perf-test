/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
namespace Act
{
    /**
     * Interface for activation function using half precision (16-bit floating point).
     */
    interface IHVecActFn
    {
        Tin::HCoopVector<Z> eval<let Z : uint>(Tin::HCoopVector<Z> x);
    };

    // Implementations of half precision activation functions.

    struct HNone : IHVecActFn
    {
        Tin::HCoopVector<Z> eval<let Z : uint>(Tin::HCoopVector<Z> x) { return x; }
    };

    struct HLinear : IHVecActFn
    {
        float16_t c;

        __init(float16_t c = 1.h) { this.c = c; }

        Tin::HCoopVector<Z> eval<let Z : uint>(Tin::HCoopVector<Z> x) { return Tin::HCoopVector<Z>(c) * x; }
    };

    struct HReLU : IHVecActFn
    {
        Tin::HCoopVector<Z> eval<let Z : uint>(Tin::HCoopVector<Z> x) { return max(x, Tin::HCoopVector<Z>(0.h)); }
    };

    struct HFixedLeakyReLU : IHVecActFn
    {
        Tin::HCoopVector<Z> eval<let Z : uint>(Tin::HCoopVector<Z> x) { return max(x, Tin::HCoopVector<Z>(0.01h) * x); }
    };

    struct HExponential3 : IHVecActFn
    {
        Tin::HCoopVector<Z> eval<let Z : uint>(Tin::HCoopVector<Z> x) { return exp(x - Tin::HCoopVector<Z>(3.h)); }
    };

    struct HHardGELU : IHVecActFn
    {
        Tin::HCoopVector<Z> eval<let Z : uint>(Tin::HCoopVector<Z> x)
        {
            let vec0 = Tin::HCoopVector<Z>(0.h);
            let vec1_5 = Tin::HCoopVector<Z>(1.5h);
            let vec3 = Tin::HCoopVector<Z>(3.h);

            return x * min(max(x + vec1_5, vec0), vec3) / vec3;
        }
    };
}
