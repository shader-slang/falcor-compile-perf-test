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
import Internal.Scene2.Geometry.TriangleGeo.TriangleGeoSharedTypes;

struct TriangleGeoData
{
    StructuredBuffer<TriangleGeoDesc> descs;
    StructuredBuffer<TriangleStaticVertex> vertices;
    StructuredBuffer<uint> indices;
}

struct TriangleAnimationVertexData
{
    RWStructuredBuffer<TriangleStaticVertex> vertices;
}
