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
 * This file contains implementations of various methods for randomizing
 * low-discrepency points.
 *
 * Pending switching to slang generics, each such method should provide a
 * randomizeSample() overload along the lines of the ones in this file.
 *
 * Note that the randomizers both take and receive 32-bit integers that
 * represent samples in [0,1) in fixed point.
 */

struct KolligKellerScrambler
{
    uint hash;
};

/**
 * Creates Kollig&Keller scrambler.
 * @param[in] hash Initial hash. This must have high entropy (i.e., seed it with a random number generator).
 */
KolligKellerScrambler createKolligKellerScrambler(uint hash)
{
    KolligKellerScrambler s;
    s.hash = hash;
    return s;
}

uint randomizeSample(inout KolligKellerScrambler kks, uint v)
{
    v ^= kks.hash;

    // Get ready for the next dimension.
    // TODO: Where did the hash functions below come from? Switched back to standard LCG for now as it has known properties.
    /*
    kks.hash ^= kks.hash * 0xe98236fe;
    kks.hash ^= kks.hash << 6;
    kks.hash ^= kks.hash * 0x7137f10c;
    kks.hash ^= kks.hash << 3;
    */
    /*
    kks.hash = kks.hash * 1103515245u + 12345u;
    */
    // Use the standard LCG from "Numerical Recipes" to get the next random hash.
    // TODO: We shouldn't use exactly the same parameters as LCG.slang, just in case someone
    // tries to scramble the LCG numbers (not likely to happen, but you never know).
    const uint A = 1664525u;
    const uint C = 1013904223u;
    kks.hash = (A * kks.hash + C);

    return v;
}

struct OwenScrambler
{};

uint randomizeSample(OwenScrambler o, uint v)
{
    v = reversebits(v);
    v ^= v * 0x782616af;
    v ^= v << 3;
    v ^= v * 0xf93ba102;
    v ^= v << 5;
    v ^= v * 0x0527ec21;
    v ^= v << 2;
    return reversebits(v);
}

struct CranleyPattersonRotater
{
    uint offset;
};

CranleyPattersonRotater createCranleyPattersonRotater(float offset)
{
    CranleyPattersonRotater cr;
    cr.offset = uint(frac(offset) * 0x1p32);
    return cr;
}

uint randomizeSample(CranleyPattersonRotater cpr, uint v)
{
    // Wrap-around just works since we're doing this in fixed point.
    return v + cpr.offset;
}
