/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#include "Scene/SDFs/SDFGridNoDefines.slangh"

__exported import Scene.SDFs.SDFGridHitData;

 /** The scene sets the SCENE_SDF_GRID_IMPLEMENTATION define to select what SDF grid implementation should be used.
     This code typedefs the chosen type to the type 'SDFGrid'.
     All SDF grid implementation adhere to the same interface.
 */

#if SCENE_SDF_GRID_IMPLEMENTATION == SCENE_SDF_GRID_IMPLEMENTATION_NDSDF
import Scene.SDFs.NormalizedDenseSDFGrid.NDSDFGrid;
typedef NDSDFGrid SDFGrid;
#elif SCENE_SDF_GRID_IMPLEMENTATION == SCENE_SDF_GRID_IMPLEMENTATION_SVS
import Scene.SDFs.SparseVoxelSet.SDFSVS;
typedef SDFSVS SDFGrid;
#elif SCENE_SDF_GRID_IMPLEMENTATION == SCENE_SDF_GRID_IMPLEMENTATION_SBS
import Scene.SDFs.SparseBrickSet.SDFSBS;
typedef SDFSBS SDFGrid;
#elif SCENE_SDF_GRID_IMPLEMENTATION == SCENE_SDF_GRID_IMPLEMENTATION_SVO
import Scene.SDFs.SparseVoxelOctree.SDFSVO;
typedef SDFSVO SDFGrid;
#else
// If no SDFs are present, define an SDFGrid dummy so that compilation doesn't fail.
import Scene.SDFs.SDFGridBase;
struct SDFGridDummy : SDFGridBase, ISDFGrid
{
    bool intersectSDF(const float3 rayOrigLocal, const float3 rayDirLocal, const float tMin, const float tMax, const uint primitiveID, out float t, out SDFGridHitData hitData) { t = -1.0f; hitData = {}; return false; }
    bool intersectSDFAny(const float3 rayOrigLocal, const float3 rayDirLocal, const float tMin, const float tMax, const uint primitiveID) { return false; }
    void decodeHit(const SDFGridHitData hitData, const float3 hitPosition, out float3 gradient) { gradient = float3(0.0f); };
};
typedef SDFGridDummy SDFGrid;
#endif
