/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import Utils.Math.BitTricks;

ByteAddressBuffer testData;
RWStructuredBuffer<uint> result;

uint pack(uint2 v)
{
    return (v.y << 16) | v.x;
}

[numthreads(256, 1, 1)]
void testBitInterleave(uint3 threadId: SV_DispatchThreadID)
{
    const uint idx = threadId.x;
    const uint data = testData.Load(idx * 4); // data = [ y15..y0 x15..x0 ]

    // Use hi/lo words of each dword in the test set as input:
    const uint2 bits = uint2(data & 0xffff, data >> 16); // bits.x = [ 0..0 x15..x0 ], bits.y = [ 0..0 y15..y0 ]
    const uint interleavedBits = interleave_32bit(bits); // interleavedBits = [y15 x15 .. y0 x0]

    // Test bit interleave.
    // Put bogus data in the unused bits to test that it doesn't change the result.
    result[5 * idx + 0] = interleave_32bit(bits ^ 0xbeef0000);
    result[5 * idx + 1] = interleave_16bit(bits ^ 0xbeefed00);

    // Test bit de-interleave.
    // These functions expect zeros for the unused bits.
    result[5 * idx + 2] = pack(deinterleave_16bit(interleavedBits & 0x0000ffff));
    result[5 * idx + 3] = pack(deinterleave_8bit(interleavedBits & 0x000000ff));

    // Test bit de-interleave of two sequences (bits 0-3 and bits 8-11).
    // Note: the result is placed in separate 16-bit words, so shift down by 8 bits.
    uint2 temp = deinterleave_2x8bit(interleavedBits & 0x00ff00ff);
    result[5 * idx + 4] = pack(((temp >> 8) & 0x0f00) | (temp & 0xf));
}
