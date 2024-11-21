/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
struct ParamBlock
{
    uint virtualBricksPerAxis;
    Buffer<uint> indirectionBuffer;
    RWTexture3D<uint> indirectionTexture;
};

ParameterBlock<ParamBlock> gParamBlock;

[numthreads(4, 4, 4)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID)
{
    if (any(dispatchThreadID >= gParamBlock.virtualBricksPerAxis))
        return;

    const uint3 virtualBrickCoords = dispatchThreadID;
    uint virtualBrickID = virtualBrickCoords.x + gParamBlock.virtualBricksPerAxis * (virtualBrickCoords.y + gParamBlock.virtualBricksPerAxis * virtualBrickCoords.z);

    gParamBlock.indirectionTexture[virtualBrickCoords] = gParamBlock.indirectionBuffer[virtualBrickID];
}
