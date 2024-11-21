/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
StructuredBuffer<int> testType;
StructuredBuffer<int2> testValue;
StructuredBuffer<float3> data;
RWStructuredBuffer<int> resultsInt;
RWStructuredBuffer<float2> resultsFloat;

[anyValueSize(64)]
interface ITestInterface
{
    /// Returns a value stored in the class, won't have a default implementation
    int getStoredValue();
    /// This binary op will have a default implementation, returning the lhs
    float getBinaryOp(float lhs, float rhs);
    /// This binary op will have a default implementation, returning the v
    float getUnaryOp(float v);
};

/// Implements the default implementations, does *NOT* implement the interface.
struct TestInterfaceBase
{
    int value0; ///< This tests if members of base (value0) are in memory before members of inherited class (value1)
    float getBinaryOp(float lhs, float rhs) { return lhs; }
    float getUnaryOp(float v) { return v; }
};

/// Implements all three explicitly
struct TestV0SubNeg : TestInterfaceBase, ITestInterface
{
    int value1;
    int getStoredValue() { return (value0 - value1); }
    float getBinaryOp(float lhs, float rhs) { return lhs - rhs; }
    float getUnaryOp(float v) { return -v; }
}

/// Implements only the mandatory, leaving the other two at defaults
struct TestV1DefDef : TestInterfaceBase, ITestInterface
{
    int value1;
    int getStoredValue() { return (value0 - value1) + 1; }
}

/// Implements mandatory and unary
struct TestV2DefNeg : TestInterfaceBase, ITestInterface
{
    int value1;
    int getStoredValue() { return (value0 - value1) + 2; }
    float getUnaryOp(float v) { return -v; }
}

/// Implements mandatory and binary
struct TestV3SumDef : TestInterfaceBase, ITestInterface
{
    int value1;
    int getStoredValue() { return (value0 - value1) + 3; }
    float getBinaryOp(float lhs, float rhs) { return lhs + rhs; }
}

#ifdef COMPILE_WITH_ERROR
/// Implements binary, unary, but not mandatory, should fail to compile
struct TestDefSumNeg : TestInterfaceBase, ITestInterface
{
    float getBinaryOp(float lhs, float rhs) { return lhs + rhs; }
    float getUnaryOp(float v) { return -v; }
}
#endif

ITestInterface createTestClassManual(int type, int2 value)
{
    switch (type)
    {
    case 0:
    {
        TestV0SubNeg result;
        result.value0 = value[0];
        result.value1 = value[1];
        return result;
    }
    case 1:
    {
        TestV1DefDef result;
        result.value0 = value[0];
        result.value1 = value[1];
        return result;
    }
    case 2:
    {
        TestV2DefNeg result;
        result.value0 = value[0];
        result.value1 = value[1];
        return result;
    }
    case 3:
    {
        TestV3SumDef result;
        result.value0 = value[0];
        result.value1 = value[1];
        return result;
    }
#ifdef COMPILE_WITH_ERROR
    case 4:
    {
        TestDefSumNeg result;
        return result;
    }
#endif
    default:
    {
        TestV3SumDef result;
        result.value0 = -65536;
        result.value1 = -65536;
        return result;
    }
    }
}

ITestInterface createTestClassConformance(int type, int2 value)
{
    return createDynamicObject<ITestInterface, int2>(type, value);
}

[numthreads(16, 1, 1)]
void testInheritanceManual(uint3 threadID: SV_DispatchThreadID)
{
    uint i = threadID.x;
    if (i >= NUM_TESTS)
        return;

    ITestInterface test = createTestClassManual(testType[i], testValue[i]);
    resultsInt[i] = test.getStoredValue();
    resultsFloat[i][0] = test.getBinaryOp(data[i][0], data[i][1]);
    resultsFloat[i][1] = test.getUnaryOp(data[i][2]);
}

[numthreads(16, 1, 1)]
void testInheritanceConformance(uint3 threadID: SV_DispatchThreadID)
{
    uint i = threadID.x;
    if (i >= NUM_TESTS)
        return;

    ITestInterface test = createTestClassConformance(testType[i], testValue[i]);
    resultsInt[i] = test.getStoredValue();
    resultsFloat[i][0] = test.getBinaryOp(data[i][0], data[i][1]);
    resultsFloat[i][1] = test.getUnaryOp(data[i][2]);
}
