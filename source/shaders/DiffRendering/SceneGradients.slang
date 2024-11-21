/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#include "Utils/NVAPI.slangh"

__exported import DiffRendering.SharedTypes;

struct SceneGradients
{
    uint gradDim[(uint)GradientType::Count];
    uint hashSize[(uint)GradientType::Count];

    // Temporary buffers for keeping gradients before aggregating them.
    RWByteAddressBuffer tmpGrads[(uint)GradientType::Count];

    uint getGradDim(GradientType gradType) { return gradDim[(uint)gradType]; }

    uint getHashSize(GradientType gradType) { return hashSize[(uint)gradType]; }

    float getGrad(GradientType gradType, uint gradIndex, uint hashIndex)
    {
        uint index = hashIndex * gradDim[(uint)gradType] + gradIndex;
        return asfloat(tmpGrads[(uint)gradType].Load(index * 4));
    }

    void atomicAddGrad(GradientType gradType, uint gradIndex, uint hashIndex, float value)
    {
        if (gradIndex < gradDim[(uint)gradType])
        {
            uint index = hashIndex * gradDim[(uint)gradType] + gradIndex;
            tmpGrads[(uint)gradType].InterlockedAddF32(index * 4, value);
        }
    }
};

ParameterBlock<SceneGradients> gSceneGradients;

#ifdef SCENE_GRADIENTS_BLOCK
void main() {}
#endif
