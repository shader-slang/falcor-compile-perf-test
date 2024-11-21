/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import Utils.Attributes;
import Rendering.Materials.TexLODHelpers;
__exported import Scene.Shading;

cbuffer DxrPerFrame
{
    uint rayTypeCount;
};

/** Convenience wrapper for TraceRay() that sets the Scene parameters internally.
    \param[in] rayFlags Ray flags. Set to RAY_FLAG_NONE if not used.
    \param[in] instanceInclusionMask Instance inclusion mask. Set to 0xff if not needed.
    \param[in] rayType The ray type index.
    \param[in] missIndex The miss shader index.
    \param[in] ray The ray description.
    \param[in,out] payload The payload struct.
*/
void TraceRayScene<payload_t>(
    uint rayFlags,
    uint instanceInclusionMask,
    uint rayType,
    uint missIndex,
    RayDesc ray,
    inout payload_t payload)
{
    TraceRay(gScene.rtAccel, rayFlags, instanceInclusionMask, rayType, rayTypeCount, missIndex, ray, payload);
}

/** Return the global geometry instance ID in the scene from a DXR 1.0 hit group shader.
    This is just a convenience function that internally uses the DXR system values.
    Note: This function can only be called from a closestHit, anyHit, or intersection shader.
    \return Global geometry instance ID.
*/
GeometryInstanceID getGeometryInstanceID()
{
    return GeometryInstanceID(InstanceID(), GeometryIndex());
}

/** Returns interpolated vertex attributes in a ray tracing hit program.
    \param[in] instanceID Global geometry instance ID.
    \param[in] triangleIndex Index of the triangle in the current mesh (= PrimitiveIndex()).
    \param[in] attribs Intersection attributes provided by DXR.
    \return Interpolated vertex attributes.
*/
VertexData getVertexData(GeometryInstanceID instanceID, uint triangleIndex, BuiltInTriangleIntersectionAttributes attribs)
{
    float3 barycentrics = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y, attribs.barycentrics.x, attribs.barycentrics.y);
    return gScene.getVertexData(instanceID, triangleIndex, barycentrics);
}

/** Returns interpolated vertex attributes in a ray tracing hit program.
    \param[in] instanceID Global geometry instance ID.
    \param[in] triangleIndex Index of the triangle in the current mesh (= PrimitiveIndex()).
    \param[in] attribs Intersection attributes provided by DXR.
    \param[out] Unpacked fetched vertices which can be used for further computations involving individual vertices.
    \return Interpolated vertex attributes.
*/
VertexData getVertexData(GeometryInstanceID instanceID, uint triangleIndex, BuiltInTriangleIntersectionAttributes attribs, out float3 barycentrics, out StaticVertexData vertices[3])
{
    barycentrics = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y, attribs.barycentrics.x, attribs.barycentrics.y);
    return gScene.getVertexData(instanceID, triangleIndex, barycentrics, vertices);
}

/** Returns interpolated vertex attributes in a ray tracing hit program when ray cones are used for texture LOD.
    \param[in] instanceID Global geometry instance ID.
    \param[in] triangleIndex Index of the triangle in the current mesh (= PrimitiveIndex()).
    \param[in] attribs Intersection attributes provided by DXR.
    \return Interpolated vertex attributes.
*/
VertexData getVertexDataRayCones(GeometryInstanceID instanceID, uint triangleIndex, BuiltInTriangleIntersectionAttributes attribs)
{
    float3 barycentrics = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y, attribs.barycentrics.x, attribs.barycentrics.y);
    return gScene.getVertexDataRayCones(instanceID, triangleIndex, barycentrics);
}

/** Returns interpolated position on a triangle in world space for the previous frame.
    \param[in] instanceID Global geometry instance ID.
    \param[in] triangleIndex Index of the triangle in the current mesh (= PrimitiveIndex()).
    \param[in] attribs Intersection attributes provided by DXR.
    \return Interpolated position in world space for the previous frame.
*/
float3 getPrevPosW(GeometryInstanceID instanceID, uint triangleIndex, BuiltInTriangleIntersectionAttributes attribs)
{
    float3 barycentrics = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y, attribs.barycentrics.x, attribs.barycentrics.y);
    return gScene.getPrevPosW(instanceID, triangleIndex, barycentrics);
}
