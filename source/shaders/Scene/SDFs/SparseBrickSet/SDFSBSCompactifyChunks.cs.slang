/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
struct ParamBlock
{
    uint chunkCount;
    ByteAddressBuffer chunkIndirection;
    ByteAddressBuffer chunkValidity;
    ByteAddressBuffer chunkCoords;
    RWByteAddressBuffer compactedChunkCoords;
};

ParameterBlock<ParamBlock> gParamBlock;

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= gParamBlock.chunkCount) return;

    const uint readAddress = index << 2;
    const bool valid = gParamBlock.chunkValidity.Load(readAddress) != 0;
    if (!valid) return;

    const uint compactedIndex = gParamBlock.chunkIndirection.Load(readAddress);
    const uint3 coords = gParamBlock.chunkCoords.Load3(3 * readAddress);

    gParamBlock.compactedChunkCoords.Store3(3 * (compactedIndex << 2), coords);
}
