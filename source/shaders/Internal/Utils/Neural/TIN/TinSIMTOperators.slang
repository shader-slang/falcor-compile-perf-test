/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/

// These are vector operators for use when native operators (via the VK_NV_COOP_VECTOR) are not supported

static Tin::HCoopVector<Z> operator +<let Z : uint>(Tin::HCoopVector<Z> a, Tin::HCoopVector<Z> b)
{
    Tin::HCoopVector<Z> r;
    [[unroll]]
    for (uint i = 0; i < Z; i++)
    {
        r[i] = a[i] + b[i];
    }
    return r;
}

static Tin::HCoopVector<Z> operator -<let Z : uint>(Tin::HCoopVector<Z> a, Tin::HCoopVector<Z> b)
{
    Tin::HCoopVector<Z> r;
    [unroll]
    for (uint i = 0; i < Z; i++)
    {
        r[i] = a[i] - b[i];
    }
    return r;
}

Tin::HCoopVector<Z> operator*<let Z : uint>(Tin::HCoopVector<Z> a, Tin::HCoopVector<Z> b)
{
    Tin::HCoopVector<Z> r;
    [unroll]
    for (uint i = 0; i < Z; i++)
    {
        r[i] = a[i] * b[i];
    }
    return r;
}

Tin::HCoopVector<Z> operator /<let Z : uint>(Tin::HCoopVector<Z> a, Tin::HCoopVector<Z> b)
{
    Tin::HCoopVector<Z> r;
    [unroll]
    for (uint i = 0; i < Z; i++)
    {
        r[i] = a[i] / b[i];
    }
    return r;
}

// Misc math functions
Tin::HCoopVector<Z> min<let Z : uint>(Tin::HCoopVector<Z> a, Tin::HCoopVector<Z> b)
{
    Tin::HCoopVector<Z> r;
    [unroll]
    for (uint i = 0; i < Z; i++)
    {
        r[i] = min(a[i], b[i]);
    }
    return r;
}

Tin::HCoopVector<Z> max<let Z : uint>(Tin::HCoopVector<Z> a, Tin::HCoopVector<Z> b)
{
    Tin::HCoopVector<Z> r;
    [unroll]
    for (uint i = 0; i < Z; i++)
    {
        r[i] = max(a[i], b[i]);
    }
    return r;
}

Tin::HCoopVector<Z> exp<let Z : uint>(Tin::HCoopVector<Z> a)
{
    Tin::HCoopVector<Z> r;
    [unroll]
    for (uint i = 0; i < Z; i++)
    {
        r[i] = exp(a[i]);
    }
    return r;
}
