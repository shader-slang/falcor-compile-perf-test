/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import Utils.Attributes;

/** Parameter block for testing root descriptors.

    The root descriptor is mixed in with other non-root resources to ensure
    it doesn't break access to those.
*/
struct TestBlock
{
    float c0;
    ByteAddressBuffer bufA[2];
    RWBuffer<float> bufB[3];

#if USE_UAV
    [root]
    RWByteAddressBuffer testBuffer; // Buffer bound to root descriptor (UAV)
#else
    [root]
    ByteAddressBuffer testBuffer; // Buffer bound to root descriptor (SRV)
#endif

    Buffer<uint> bufC[4];
};
