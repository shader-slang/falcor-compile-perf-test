/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/

/**
 * Interface for activation function using full precision (32-bit floating point).
 */
interface IActFn
{
    float eval(float x);
};

/**
 * Interface for activation function using half precision (16-bit floating point).
 */
interface IHActFn
{
    float16_t eval(float16_t x);
};

/**
 * Interface for activation function backpropagation using half precision (16-bit floating point).
 */
interface IHActFnBackward
{
    float16_t backward(float16_t y, float16_t x = 1.h);
};

/**
 * Namespace for activation functions so that we can use their short names
 * such as `Exp` and `None` without clashing with other code.
 */
namespace Act
{
    // Implementations of full precision activation functions.

    struct None : IActFn
    {
        __init() {}
        float eval(float x) { return x; }
    }

    struct Linear : IActFn
    {
        float c;
        __init(float c = 1.f) { this.c = c; }
        float eval(float x) { return c * x; }
    }

    struct BinaryStep : IActFn
    {
        __init() {}
        float eval(float x) { return step(0.f, x); }
    }

    struct ReLU : IActFn
    {
        __init() {}
        float eval(float x) { return max(x, 0.f); }
    };

    struct LeakyReLU : IActFn
    {
        float a;
        __init(float a = 0.01f) { this.a = a; }
        float eval(float x) { return x < 0.f ? a * x : x; }
    };

    struct ELU : IActFn
    {
        float a;
        __init(float a = 1.f) { this.a = a; }
        float eval(float x) { return x < 0.f ? a * (exp(x) - 1.f) : x; }
    };

    struct Sigmoid : IActFn
    {
        __init() {}
        float eval(float x) { return 1.f / (1.f + exp(-x)); }
    };

    struct Swish : IActFn
    {
        __init() {}
        float eval(float x) { return x / (1.f - exp(-x)); }
    };

    struct HardSwish : IActFn
    {
        __init() {}
        float eval(float x) { return x * clamp(x + 3.f, 0.f, 6.f) / 6.f; }
    };

    struct HardGELU : IActFn
    {
        __init() {}
        float eval(float x) { return x * clamp(x + 1.5f, 0.f, 3.f) / 3.f; }
    };

    struct Tanh : IActFn
    {
        __init() {}
        float eval(float x) { return 2.f / (1.f + exp(-2.f * x)) - 1.f; }
    };

    /**
     * The Softplus function is a smooth approximation of ReLU.
     * For numerical stability the implementation reverts to a linear function when beta * x > threshold.
     */
    struct Softplus : IActFn
    {
        float beta, threshold;
        __init(float beta = 1.f, float threshold = 20.f)
        {
            this.beta = beta;
            this.threshold = threshold;
        }
        float eval(float x) { return beta * x > threshold ? x : log(1.f + exp(beta * x)) / beta; }
    };

    struct Exp2 : IActFn
    {
        __init() {}
        float eval(float x) { return exp2(x) - 1.f; }
    };

    struct Exp : IActFn
    {
        __init() {}
        float eval(float x) { return exp(x) - 1.f; }
    };

    struct Exponential3 : IActFn
    {
        float offset;
        __init(float offset = -3.f) { this.offset = offset; }
        float eval(float x) { return exp(x + offset); }
    };

    // Implementations of half precision activation functions.

    struct HNone : IHActFn
    {
        __init() {}
        float16_t eval(float16_t x) { return x; }
    }

    struct HLinear : IHActFn
    {
        float16_t c;
        __init(float16_t c = 1.h) { this.c = c; }
        float16_t eval(float16_t x) { return c * x; }
    }

    struct HBinaryStep : IHActFn
    {
        __init() {}
        float16_t eval(float16_t x) { return step(0.h, x); }
    }

    struct HReLU : IHActFn
    {
        __init() {}
        float16_t eval(float16_t x) { return max(x, 0.h); }
    };

    struct HLeakyReLU : IHActFn
    {
        float16_t a;
        __init(float16_t a = 0.01h) { this.a = a; }
        float16_t eval(float16_t x) { return x < 0.h ? a * x : x; }
    };

    struct HELU : IHActFn
    {
        float16_t a;
        __init(float16_t a = 1.h) { this.a = a; }
        float16_t eval(float16_t x) { return x < 0.h ? a * (exp(x) - 1.h) : x; }
    };

    struct HSigmoid : IHActFn
    {
        __init() {}
        float16_t eval(float16_t x) { return 1.h / (1.h + exp(-x)); }
    };

    struct HSwish : IHActFn
    {
        __init() {}
        float16_t eval(float16_t x) { return x / (1.h - exp(-x)); }
    };

    struct HHardSwish : IHActFn
    {
        __init() {}
        float16_t eval(float16_t x) { return x * clamp(x + 3.h, 0.h, 6.h) / 6.h; }
    };

    struct HHardGELU : IHActFn
    {
        __init() {}
        float16_t eval(float16_t x) { return x * clamp(x + 1.5h, 0.h, 3.h) / 3.h; }
    };

    struct HTanh : IHActFn
    {
        __init() {}
        float16_t eval(float16_t x) { return 2.h / (1.h + exp(-2.h * x)) - 1.h; }
    };

    /**
     * The Softplus function is a smooth approximation of ReLU.
     * For numerical stability the implementation reverts to a linear function when beta * x > threshold.
     */
    struct HSoftplus : IHActFn
    {
        float16_t beta, threshold;
        __init(float16_t beta = 1.h, float16_t threshold = 20.h)
        {
            this.beta = beta;
            this.threshold = threshold;
        }
        float16_t eval(float16_t x) { return beta * x > threshold ? x : log(1.h + exp(beta * x)) / beta; }
    };

    struct HExp2 : IHActFn
    {
        __init() {}
        float16_t eval(float16_t x) { return exp2(x) - 1.h; }
    };

    struct HExp : IHActFn
    {
        __init() {}
        float16_t eval(float16_t x) { return exp(x) - 1.h; }
    };

    struct HExponential3 : IHActFn
    {
        __init() {}
        float16_t eval(float16_t x) { return exp(x - 3.h); }
    };

    // Fixed versions of above stateful activations functions.
    // This is a workaround for Slang issue with generics and stateful activations.
    // See https://gitlab-master.nvidia.com/nvresearch-gfx/Tools/Falcor/-/issues/1613

    struct HFixedLinear : IHActFn
    {
        __init() {}
        float16_t eval(float16_t x) { return x; }
    }

    struct HFixedLeakyReLU : IHActFn
    {
        __init() {}
        float16_t eval(float16_t x) { return x < 0.h ? 0.01h * x : x; }
    };

    struct HFixedELU : IHActFn
    {
        __init() {}
        float16_t eval(float16_t x) { return x < 0.h ? (exp(x) - 1.h) : x; }
    };

    struct HFixedSoftplus : IHActFn
    {
        __init() {}
        float16_t eval(float16_t x) { return x > 20.h ? x : log(1.h + exp(x)); }
    };

    // Implementations of half precision activation function backpropagation.

    extension HNone : IHActFnBackward
    {
        float16_t backward(float16_t yd, float16_t x)
        {
            return yd;
        }
    }

    extension HLinear : IHActFnBackward
    {
        float16_t backward(float16_t yd, float16_t x)
        {
            return yd * c;
        }
    }

    extension HReLU : IHActFnBackward
    {
        float16_t backward(float16_t yd, float16_t x)
        {
            return x > 0.h ? yd : 0.h;
        }
    }

    extension HLeakyReLU : IHActFnBackward
    {
        float16_t backward(float16_t yd, float16_t x)
        {
            return x > 0.h ? yd : a * yd;
        }
    }

    extension HHardGELU : IHActFnBackward
    {
        float16_t backward(float16_t yd, float16_t x)
        {
            return x > -1.5h ? x < 1.5h ? 0.5h * yd + float16_t(2.f / 3.f) * x * yd : yd : 0.h;
        }
    }

    extension HFixedLeakyReLU : IHActFnBackward
    {
        float16_t backward(float16_t yd, float16_t x)
        {
            return x > 0.h ? yd : 0.01h * yd;
        }
    }
} // namespace Act
