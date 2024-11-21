/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/

/** Packed version of PolymorphicLight (defined in PolymorphicLight.slang).
*/
struct PackedPolymorphicLight
{
    uint header;
    uint3 data0;
    uint4 data1;
};

/** Packed version of SurfaceData (defined in SurfaceData.slang).
*/
struct PackedSurfaceData
{
    uint3 position;
    uint viewDepth;
    uint normal;
    uint weights;
    uint2 _pad;
};
