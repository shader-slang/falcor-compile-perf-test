/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import Scene.Raster;

VSOut vsMain(VSIn vIn)
{
    return defaultVS(vIn);
}

void psMain(VSOut vsOut, uint triangleIndex: SV_PrimitiveID) : SV_TARGET
{
#if USE_ALPHA_TEST
    let lod = ImplicitLodTextureSampler();
    if (alphaTest(vsOut, triangleIndex, lod))
        discard;
#endif
}
