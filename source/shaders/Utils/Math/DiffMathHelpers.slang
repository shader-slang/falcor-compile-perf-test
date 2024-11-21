/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/

/**
 * Infinitesimal functions to extract gradient.
 */
[Differentiable]
[PreferRecompute]
[ForceInline]
float infinitesimal(float x)
{
    return x - detach(x);
}

[Differentiable]
[PreferRecompute]
[ForceInline]
__generic<let N : int> vector<float, N> infinitesimal(vector<float, N> x)
{
    return x - detach(x);
}
