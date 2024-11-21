/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import DiffRendering.SceneGradients;
import DiffRendering.DiffDebugParams;
import Scene.Material.MaterialParamLayout;
import Utils.Math.HashUtils;

// Avoid compilation warnings if not running in a differentiable mode.
#ifndef DIFF_MODE
#define DIFF_MODE 0
#endif

// Simple hash function.
uint hashFunction(uint index, uint hashSize)
{
    return jenkinsHash(index) % hashSize;
}

/**
 * The wrapper function has custom derivatives for setting up the data flow of scene gradients during backpropagation.
 * It indicates where to accumulate the gradients by providing `offset` in the gradient buffer.
 */
struct GradientIOWrapper
{
    GradientType gradType;
    uint baseOffset;
    uint hashIndex;

    __init(GradientType _gradType, uint _baseOffset, uint _hashIndex)
    {
        this.gradType = _gradType;
        this.baseOffset = _baseOffset;
        this.hashIndex = _hashIndex;
    }

    [ForwardDerivative(fwd_d_getFloat)]
    [BackwardDerivative(bwd_d_getFloat)]
    float getFloat(float val, uint offset) { return val; }

    [TreatAsDifferentiable]
    DifferentialPair<float> fwd_d_getFloat(DifferentialPair<float> dpVal, uint offset) { return diffPair(dpVal.p, 0.f); }

    [TreatAsDifferentiable]
    void bwd_d_getFloat(inout DifferentialPair<float> dpVal, uint offset, float dOut)
    {
        offset += baseOffset;
        gSceneGradients.atomicAddGrad(gradType, offset, hashIndex, dOut);
    }

    [ForwardDerivative(fwd_d_getFloats)]
    [BackwardDerivative(bwd_d_getFloats)]
    __generic<let N : int> vector<float, N> getFloat(vector<float, N> val, uint offset) { return val; }

    [TreatAsDifferentiable]
    __generic<let N : int> DifferentialPair<vector<float, N>> fwd_d_getFloats(DifferentialPair<vector<float, N>> dpVal, uint offset)
    {
        vector<float, N> d;
        [ForceUnroll]
        for (int i = 0; i < N; i++)
            d[i] = 0.f;

#if DIFF_MODE == 2 // ForwardDiffDebug
        if (gDiffDebug.varType == DiffVariableType::Material && baseOffset == gDiffDebug.id.x * 20 && offset == gDiffDebug.offset)
        {
            [ForceUnroll]
            for (int i = 0; i < N; i++)
                d[i] = gDiffDebug.grad[i];
        }
#endif

        return diffPair(dpVal.p, d);
    }

    [TreatAsDifferentiable]
    __generic<let N : int> void bwd_d_getFloats(inout DifferentialPair<vector<float, N>> dpVal, uint offset, vector<float, N> dOut)
    {
        offset += baseOffset;
        [ForceUnroll]
        for (uint i = 0; i < N; i++)
            gSceneGradients.atomicAddGrad(gradType, offset + i, hashIndex, dOut[i]);
    }

    [Differentiable]
    float getFloat(float val, const MaterialParamLayoutEntry entry) { return getFloat(val, entry.offset); }

    [Differentiable]
    __generic<let N : int> vector<float, N> getFloat(vector<float, N> val, const MaterialParamLayoutEntry entry)
    {
        return getFloat(val, entry.offset);
    }
};
