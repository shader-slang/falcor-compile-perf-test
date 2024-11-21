/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
RWStructuredBuffer<uint> result;

struct Test1
{
    struct A<let N : uint = 2>
    {
        static uint num() { return N; }
    };

    static uint num() { return A<2>::num(); }

    typealias B = A<2>;
};

struct Test2
{};

extension Test2
{
    struct A<let N : uint = 2>
    {
        static uint num() { return N; }
    };

    static uint num()
    {
        return A<2>::num(); // ERROR
    }

    typealias B = A<2>;
};

[numthreads(1, 1, 1)]
void main(uint3 threadID: SV_DispatchThreadID)
{
    result[0] = Test1::num();
    result[1] = Test1::B::num();
    result[2] = Test1::A<2>::num(); // ERROR

    result[3] = Test2::num(); // ERROR
    result[4] = Test2::B::num();
    result[5] = Test2::A<2>::num(); // ERROR
}
