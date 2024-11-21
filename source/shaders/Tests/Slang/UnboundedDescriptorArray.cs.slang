/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
struct Resources
{
    Texture2D textures[];

    float get(uint id, int2 pos) { return textures[id].Load(int3(pos, 0)).x; }
};

ParameterBlock<Resources> resources;
RWStructuredBuffer<float> result;

[numthreads(4, 1, 1)]
void main(uint3 threadID: SV_DispatchThreadID)
{
    uint i = threadID.x;
    result[i] = resources.get(i, int2(0, 0));
}
