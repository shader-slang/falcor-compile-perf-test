/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import Scene.Scene;
__exported import InteriorList;

/** Compute index of refraction for medium on the outside of the current dielectric volume.
    \param[in] interiorList Interior list.
    \param[in] materialID Material ID of intersected material.
    \param[in] entering True if material is entered, false if material is left.
    \return Index of refraction.
*/
float computeOutsideIoR(const InteriorList interiorList, const uint materialID, const bool entering)
{
    // The top element holds the material ID of currently highest priority material.
    // This is the material on the outside when entering a new medium.
    uint outsideMaterialID = interiorList.getTopMaterialID();

    if (!entering)
    {
        // If exiting the currently highest priority material, look at the next element
        // on the stack to find out what is on the outside.
        if (outsideMaterialID == materialID) outsideMaterialID = interiorList.getNextMaterialID();
    }

    // If no material, assume the default IoR for vacuum.
    if (outsideMaterialID == InteriorList::kNoMaterial) return 1.f;

    return gScene.materials.evalIoR(outsideMaterialID);
}
