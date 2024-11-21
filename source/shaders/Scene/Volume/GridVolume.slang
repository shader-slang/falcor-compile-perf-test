/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
__exported import Utils.Math.AABB;
__exported import Grid;
import GridVolumeData;

/** Grid volume.
*/
struct GridVolume
{
    /** Specifies how emission is rendered.
    */
    enum class EmissionMode
    {
        Direct      = 0,
        Blackbody   = 1,
    };

    static const uint kInvalidGrid = -1;

    GridVolumeData data;

    /** Get the world-space bounds.
    */
    AABB getBounds()
    {
        return AABB(data.boundsMin, data.boundsMax);
    }

    /** Get the emission mode.
    */
    EmissionMode getEmissionMode()
    {
        return (EmissionMode)data.flags;
    }

    /** Get the phase function anisotropy.
    */
    float getAnisotropy()
    {
        return data.anisotropy;
    }

    bool hasDensityGrid() { return data.densityGrid != kInvalidGrid; }
    bool hasEmissionGrid() { return data.emissionGrid != kInvalidGrid; }

    uint getDensityGrid() { return data.densityGrid; }
    uint getEmissionGrid() { return data.emissionGrid; }
};
