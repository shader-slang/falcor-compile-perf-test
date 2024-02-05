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
 * Pack/unpack utilities for FP8<->FP16 conversion.
 * The rounding mode is always round toward nearest even (RNE).
 * See host side reference implementation for commentary and tests.
 */

uint16_t pack_fp8_e4m3(const float16_t val, const bool saturate = false)
{
    int16_t bits = asint16(val);

    int16_t s = (bits >> 8) & 0x0080;
    int16_t e = ((bits >> 10) & 0x001f) - (15 - 7);
    int16_t m = bits & 0x03ff;

    if (e < -3)
    {
        // The absolute value is less than fMin. Convert to signed zero.
        return s;
    }
    else if (e <= 0)
    {
        // E is between -3 and 0.
        // Value is a normalized float whose magnitude is less than fMinNorm.
        // Convert to denormalized FP8.E4M3.
        m |= int16_t(0x0400);

        // Round toward nearest even.
        if (m & (int16_t(1) << (8 - e))) // Check LSB for tie breaking.
            m += (int16_t(1) << (7 - e));
        else
            m += (int16_t(1) << (7 - e)) - 1;

        // Assemble FP8 from s, e (zero) and m.
        return (s | (m >> (8 - e)));
    }
    else if (e == 0x1f - (15 - 7))
    {
        if (m == 0)
        {
            // F is +-Inf. Convert to fMax or fNan dependending on mode.
            return saturate ? (s | 0x7e) : 0x7f;
        }
        else
        {
            // F is NaN. Return the NVIDIA canonical NaN.
            return 0x7f;
        }
    }
    else
    {
        // E is greater than zero. F is a normalized float.
        // We try to convert to a normalized FP8.E4M3.

        // Round toward nearest even.
        m += (m & int16_t(0x0080)) ? int16_t(0x0040) : int16_t(0x003f);

        // Handle significand overflow.
        if (m & int16_t(0x0400))
        {
            m = 0;
            e += int16_t(1);
        }

        // Handle exponent overflow. fMax is s.1111.110
        if (e > 15 || (e == 15 && m >= int16_t(0x0380)))
        {
            return saturate ? (s | 0x7e) : 0x7f;
        }

        // Assemble FP8 from s, e and m.
        return (s | (e << 3) | (m >> 7));
    }
}

uint16_t pack_fp8_e4m3_t2(const float16_t2 val, const bool saturate = false)
{
    return pack_fp8_e4m3(val.x, saturate) | (pack_fp8_e4m3(val.y, saturate) << 8);
}

uint pack_fp8_e4m3_t4(const float16_t4 val, const bool saturate = false)
{
    return (uint)pack_fp8_e4m3_t2(val.xy, saturate) | ((uint)pack_fp8_e4m3_t2(val.zw, saturate) << 16);
}

float16_t unpack_fp8_e4m3(const uint16_t bits)
{
    int16_t s = (bits >> 7) & 0x0001;
    int16_t e = (bits >> 3) & 0x000f;
    int16_t m = bits & 0x0007;

    if (e == 0)
    {
        if (m == 0)
        {
            // Plus or minus zero.
            return asfloat16(s << 15);
        }
        else
        {
            // Denormalized number. Renormalize it.
            while (!(m & 0x0008))
            {
                m <<= 1;
                e -= int16_t(1);
            }

            e += int16_t(1);
            m &= ~int16_t(0x0008);
        }
    }
    else if (e == 15)
    {
        if (m == 0x7)
        {
            // The value is NaN. Return the NVIDIA canonical NaN.
            // TODO: Replace by 16-bit literal when https://gitlab-master.nvidia.com/slang/slang/-/issues/145 is fixed.
            // return asfloat16(int16_t(0x7fff));
            return float16_t(asfloat(0x7fffffff));
        }
    }

    // Normalized number.
    e = e + (15 - 7);
    m = m << 7;

    return asfloat16((s << 15) | (e << 10) | m);
}

float16_t2 unpack_fp8_e4m3_t2(const uint16_t bits)
{
    return { unpack_fp8_e4m3(bits), unpack_fp8_e4m3(bits >> 8) };
}

float16_t4 unpack_fp8_e4m3_t4(const uint bits)
{
    float16_t4 r;
    r.xy = unpack_fp8_e4m3_t2((uint16_t)bits);
    r.zw = unpack_fp8_e4m3_t2((uint16_t)(bits >> 16));
    return r;
    //    return { unpack_fp8_e4m3_t2((uint16_t)bits), unpack_fp8_e4m3_t2((uint16_t)(bits >> 16)) };
}
