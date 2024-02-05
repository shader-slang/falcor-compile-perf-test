/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
import Internal.Scene2.SharedTypes;
import Internal.Scene2.Geometry.GeoBase;
import Internal.Scene2.Geometry.GeoRWAccessors;
import Internal.Scene2.RWSceneGeo;

import Internal.Scene2.Animation.Skinning.SkinningTypes;

struct SkinningHandler
{
    StructuredBuffer<SkinningWeights> skinningWeights;
    StructuredBuffer<GeoRWDesc> geoRWDesc;
    StructuredBuffer<DynamicVertexData> dynamicVertexData;
    StructuredBuffer<float4x4> sceneGraphXformWorldFromInstance;
    StructuredBuffer<float4x4> sceneGraphXformInstanceFromWorld;

    StructuredBuffer<uint> indices;
    uint indexCount;
};

ParameterBlock<SkinningHandler> gSkinningHandler;

float4x4 getMatrix(SceneGraphNodeID index)
{
    return gSkinningHandler.sceneGraphXformWorldFromInstance[index.get()];
}

float4x4 getMatrixInv(SceneGraphNodeID index)
{
    return gSkinningHandler.sceneGraphXformInstanceFromWorld[index.get()];
}

float4x4 getBoneMat(SkinningWeights weights)
{
    float4x4 boneMat = getMatrix(weights.boneIDs[0]) * weights.boneWeights[0];
    boneMat += getMatrix(weights.boneIDs[1]) * weights.boneWeights[1];
    boneMat += getMatrix(weights.boneIDs[2]) * weights.boneWeights[2];
    boneMat += getMatrix(weights.boneIDs[3]) * weights.boneWeights[3];

    float4x4 bindMatrix = getMatrix(weights.bindMatrixID);
    float4x4 skeletonMatrixInv = getMatrixInv(weights.skeletonMatrixID);
    float4x4 bindMatrixInv = getMatrixInv(weights.bindMatrixID);

    boneMat = mul(boneMat, bindMatrix);
    boneMat = mul(skeletonMatrixInv, boneMat);
    boneMat = mul(bindMatrixInv, boneMat);

    return boneMat;
}

float4x4 getBoneMatIT(SkinningWeights weights)
{
    float4x4 boneMatInv = getMatrixInv(weights.boneIDs[0]) * weights.boneWeights[0];
    boneMatInv += getMatrixInv(weights.boneIDs[1]) * weights.boneWeights[1];
    boneMatInv += getMatrixInv(weights.boneIDs[2]) * weights.boneWeights[2];
    boneMatInv += getMatrixInv(weights.boneIDs[3]) * weights.boneWeights[3];

    float4x4 skeletonMatrix = getMatrix(weights.skeletonMatrixID);

    return mul(transpose(skeletonMatrix), transpose(boneMatInv));
}

[shader("compute")]
[numthreads(16, 1, 1)]
void runSkinning(uint3 threadIdx: SV_DispatchThreadID)
{
    uint tid = threadIdx.x;
    if (tid >= gSkinningHandler.indexCount)
        return;

    uint dataId = gSkinningHandler.indices[tid];

    SkinningWeights weights = gSkinningHandler.skinningWeights[dataId];
    float4x4 boneMat = getBoneMat(weights);
    float4x4 boneMatIT = getBoneMatIT(weights);

    GeoRWDesc geoDesc = gSkinningHandler.geoRWDesc[dataId];

    DynamicVertexData data = gSkinningHandler.dynamicVertexData[dataId];
    data.position = mul(boneMat, float4(data.position, 1.f)).xyz;
    data.tangent = mul(float3x3(boneMat), data.tangent);
    data.normal = mul(float3x3(boneMatIT), data.normal);

    IGeoRWAccessor geoRWAccessor = gRWScene2Geo.getGeoRWAccessor(geoDesc.geoType);
    geoRWAccessor.setDynamicVertexData(geoDesc.vertexIndex, data);
}

struct SkinningCopier
{
    StructuredBuffer<SkinningWeights> skinningWeights;
    StructuredBuffer<GeoRWDesc> geoRWDesc;
    RWStructuredBuffer<DynamicVertexData> dynamicVertexData;

    StructuredBuffer<uint> indices;

    uint indexCount;
}

ParameterBlock<SkinningCopier> gSkinningCopier;

[shader("compute")]
[numthreads(16, 1, 1)]
void runCopy(uint3 threadIdx: SV_DispatchThreadID)
{
    uint tid = threadIdx.x;
    if (tid >= gSkinningCopier.indexCount)
        return;

    uint infoIdx = gSkinningCopier.indices[tid];
    GeoRWDesc geoDesc = gSkinningCopier.geoRWDesc[infoIdx];

    IGeoRWAccessor geoRWAccessor = gRWScene2Geo.getGeoRWAccessor(geoDesc.geoType);
    gSkinningCopier.dynamicVertexData[infoIdx] = geoRWAccessor.getDynamicVertexData(geoDesc.vertexIndex);
}
