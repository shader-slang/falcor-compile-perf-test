/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
RWStructuredBuffer<uint4> result;
RWBuffer<uint> laneCount;
Buffer<uint> testData;

[numthreads(1, 1, 1)]
void testWaveGetLaneCount(uint3 threadID: SV_DispatchThreadID)
{
    laneCount[0] = WaveGetLaneCount();
}

[numthreads(256, 1, 1)]
void testWaveMatch(uint3 threadID: SV_DispatchThreadID)
{
    const uint i = threadID.x;
    if (i == 0)
        laneCount[0] = WaveGetLaneCount();

    uint value = testData[i];
    result[i] = WaveMatch(value);
}

[numthreads(256, 1, 1)]
void testWaveMinMax(uint3 threadID: SV_DispatchThreadID)
{
    const uint i = threadID.x;
    if (i == 0)
        laneCount[0] = WaveGetLaneCount();

    float value = asfloat(testData[i]);

#if CONDITIONAL == 0
    float minVal = WaveActiveMin(value);
    float maxVal = WaveActiveMax(value);
#else
    float minVal = 0.f;
    float maxVal = 0.f;

    if (value - floor(value) < 0.5f)
    {
        minVal = WaveActiveMin(value);
        maxVal = WaveActiveMax(value);
    }
#endif

    result[2 * i + 0] = asuint(minVal);
    result[2 * i + 1] = asuint(maxVal);
}

[numthreads(32, 1, 1)]
void testWaveMaxSimpleFloat(uint3 threadID: SV_DispatchThreadID)
{
    const uint i = threadID.x;

    float value = asfloat(testData[i]);
    if (value <= -2.f)
        value = WaveActiveMax(value);

    result[i] = asuint(value);
}

[numthreads(32, 1, 1)]
void testWaveMaxSimpleInt(uint3 threadID: SV_DispatchThreadID)
{
    const uint i = threadID.x;

    int value = (int)testData[i];
    if (value <= -2)
        value = WaveActiveMax(value);

    result[i] = (uint)value;
}
