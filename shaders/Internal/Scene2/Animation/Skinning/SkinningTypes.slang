/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
#pragma once
#include "Utils/HostDeviceShared.slangh"
#include "Internal/Scene2/Scene2IDs.slangh"

BEGIN_NAMESPACE(Falcor::scene2)

// Skinning transforms:
// "boneMatrices" - "skinningMatrices"
//      worldFromLocal - constructed as worldFromThis (moves) * bindFromLocal (static)
//
// Full skinning then goes (applied top to bottom, to make it sane to write, turns into right to left):
//    position in mesh (in local)   -- in local/object space of the mesh
//    meshBind (skeletonFromLocal)  -- transformed to the skeleton space (same skeleton can be used for multiple meshes)
//    bindFromLocal (rest pose joint transform) -- places the joint in rest pose when no transform is done?
//    worldFromLocal (for the bone, this gets animated) -- positions the joints (this is animated and concatenated through the skeleton)
//                                 ^^ these two create "boneMatrix"
//    localFromWorld (using skeletonMatrixId) -- this is probably root of the skeleton, i.e., the step above wants "skeletonFromLocal"
//    rather than worldFromLocal localFromSkeleon -- transform from skeleton space to local/object space of the mesh
//      -> skinned position in mesh
//    Then any parent animation of the mesh itself gets applied to rigidly transform the mesh

struct SkinningWeights
{
    SETTER_DECL
    void setWeights(float w0, float w1, float w2, float w3)
    {
        boneWeights[0] = w0;
        boneWeights[1] = w1;
        boneWeights[2] = w2;
        boneWeights[3] = w3;
    }

    SETTER_DECL
    void setIndices(SceneGraphNodeID i0, SceneGraphNodeID i1, SceneGraphNodeID i2, SceneGraphNodeID i3)
    {
        boneIDs[0] = i0;
        boneIDs[1] = i1;
        boneIDs[2] = i2;
        boneIDs[3] = i3;
    }

    float boneWeights[4];
    SceneGraphNodeID boneIDs[4];
    SceneGraphNodeID bindMatrixID;
    SceneGraphNodeID skeletonMatrixID;
};

END_NAMESPACE
