/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import Scene.Material.ShadingUtils;

/**
 * Stores differentiable parameters used for BSDF evaluation.
 */
struct DiffMaterialData : IDifferentiable
{
    static const uint kMaterialParamCount = 20;

    ShadingFrame sf;

    // Material parameters.
    float data[kMaterialParamCount];

    [Differentiable]
    __init()
    {
        sf = ShadingFrame.createIdentity();
        [ForceUnroll]
        for (uint i = 0; i < kMaterialParamCount; i++)
            data[i] = 0.f;
    }

    [Differentiable]
    float read(inout uint offset) { return data[offset++]; }

    [Differentiable]
    void read<let N : int>(out vector<float, N> value, inout uint offset)
    {
        [ForceUnroll]
        for (uint i = 0; i < N; i++)
            value[i] = read(offset);
    }

    [Differentiable]
    vector<float, N> read<let N : int>(inout uint offset)
    {
        vector<float, N> value;
        this.read(value, offset);
        return value;
    }

    [mutating]
    [Differentiable]
    void write(float value, inout uint offset) { data[offset++] = value; }

    [mutating]
    [Differentiable]
    void write<let N : int>(vector<float, N> value, inout uint offset)
    {
        [ForceUnroll]
        for (uint i = 0; i < N; i++)
            this.write(value[i], offset);
    }
};
