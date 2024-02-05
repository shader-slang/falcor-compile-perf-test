/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
import Internal.Scene2.HitInfo;
import Internal.Scene2.VertexData;
import Internal.Scene2.SharedTypes;

import Utils.Math.Ray;
struct GeoIsectAttributes
{
    uint data[8]; // Max size if 32B
}

[anyValueSize(64)]
interface IGeoIntersector
{
    associatedtype GeoHitInfo : IHitInfo;

    /// Used for the hit is recompacted with type for selection, but using the
    /// type conformances to get the result (it can also use explicit switch on type,
    /// which wouldn't go through this interface at all).
    bool getVertexData(const Ray ray, const HitInfo hitInfo, out VertexData v);
    /// Used when the hit is fully dynamic
    bool getVertexData(const Ray ray, const GeoHitInfo hitInfo, out VertexData v);
    GeoHitInfo makeHitInfo(const HitObject hitObject);
    GeoHitInfo makeHitInfo(const GeoIsectAttributes);
    // Not supported by Slang as of 0.27.8, check with @yhe
    GeoHitInfo makeHitInfo<let Flags : int>(RayQuery<Flags> rayQuery, const GeoIsectAttributes);
    GeometryType2 getType();
    // These are to be dummy for triangle-based primitives with native intersectors
    bool intersect(const Ray ray, uint geoDescOffset, out GeoIsectAttributes attr, out float t);
    // Not supported by Slang as of 0.27.8, check with @yhe
    bool intersect<let Flags : int>(RayQuery<Flags> rayQuery, out GeoIsectAttributes attr, out float t);
}

[anyValueSize(64)]
interface IGeoRWAccessor
{
    // Get DynamicVertexData for the given info. Called when geo is first introduced to create
    // a stable (non-skinned) baseline from which skinning will be calculated
    DynamicVertexData getDynamicVertexData(uint vertexIndex);
    // Set DynamicVertexData for the given info, after it has been interpolated.
    // Used by skinning before the accelleration structures are built
    void setDynamicVertexData(uint vertexIndex, DynamicVertexData data);
}
