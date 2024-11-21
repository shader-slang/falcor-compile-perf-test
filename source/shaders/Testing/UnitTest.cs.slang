/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
RWStructuredBuffer<float> result;

cbuffer TestCB
{
    int nValues;
    float scale;
};

void main()
{
    for (int i = 0; i < nValues; ++i)
    {
        result[i] = scale * i;
    }
}
