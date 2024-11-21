/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
RWStructuredBuffer<uint> result;
Buffer<uint4> buffer;

struct Data
{
    uint a;
    uint b;

    [mutating]
    void decode(uint2 packed)
    {
        a = packed.x;
        b = packed.y;
    }
};

struct DecodedData
{
    Data data;
};

DecodedData load(Buffer<uint4> buf)
{
    DecodedData d = {};
    d.data.decode(buf[0].xy);
    return d;
}

[numthreads(1, 1, 1)]
void main()
{
    DecodedData d = load(buffer);
    result[0] = d.data.a + d.data.b;
}
