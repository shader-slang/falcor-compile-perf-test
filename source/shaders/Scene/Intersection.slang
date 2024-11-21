/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import Scene.Scene;
import Scene.Displacement.DisplacementMapping;
import Scene.SDFs.SDFGrid;
import Utils.Geometry.IntersectionHelpers;
import Utils.Geometry.GeometryHelpers;
__exported import Utils.Math.Ray;
import Utils.Math.MatrixUtils;

/** Helper for intersecting rays against displaced triangle meshes.
*/
struct DisplacedTriangleMeshIntersector
{
    /** Intersection attributes.
    */
    struct Attribs
    {
        float2 barycentrics;
        float displacement;
    };

    /** Intersects a ray with a displaced triangle.
        \param[in] ray Ray in world-space.
        \param[in] instanceID Geometry instance ID.
        \param[in] primitiveIndex Primitive index.
        \param[out] attribs Intersection attributes.
        \param[out] t Intersection t.
        \return True if the ray intersects the displaced triangle.
    */
    static bool intersect(const Ray ray, const GeometryInstanceID instanceID, const uint primitiveIndex, out Attribs attribs, out float t)
    {
        attribs = {};
        t = {};

        const uint materialID = gScene.getMaterialID(instanceID);
        const uint3 indices = gScene.getIndices(instanceID, primitiveIndex);
        const StaticVertexData vertices[3] = { gScene.getVertex(indices[0]), gScene.getVertex(indices[1]), gScene.getVertex(indices[2]) };
        const float4x4 worldMat = gScene.getWorldMatrix(instanceID);

        DisplacementData displacementData;
        gScene.materials.loadDisplacementData(materialID, displacementData);

        DisplacementIntersection result;
        if (intersectDisplacedTriangle(ray, vertices, worldMat, displacementData, result))
        {
            attribs.barycentrics = result.barycentrics;
            attribs.displacement = result.displacement;
            t = result.t;
            return true;
        }

        return false;
    }
};

/** Helper for intersecting rays against curves.
*/
struct CurveIntersector
{
    static const bool kUseSphereJoints = true;

    /** Intersection attributes.
    */
    struct Attribs
    {
        float2 barycentrics;
    }

    /** Intersects a ray with a curve.
        \param[in] ray Ray in world-space.
        \param[in] instanceID Geometry instance ID.
        \param[in] primitiveIndex Primitive index.
        \param[out] attribs Intersection attributes.
        \param[out] t Intersection t.
        \return True if the ray intersects the curve.
    */
    static bool intersect(const Ray ray, const GeometryInstanceID instanceID, const uint primitiveIndex, out Attribs attribs, out float t)
    {
        attribs = {};
        t = {};

        uint v0Index = gScene.getFirstCurveVertexIndex(instanceID, primitiveIndex);
        const StaticCurveVertexData v0 = gScene.getCurveVertex(v0Index);
        const StaticCurveVertexData v1 = gScene.getCurveVertex(v0Index + 1);

        const float4x4 worldMat = gScene.getWorldMatrix(instanceID);
        const float4 sphereA = transformSphere(float4(v0.position, v0.radius), worldMat);
        const float4 sphereB = transformSphere(float4(v1.position, v1.radius), worldMat);

        const float rayLength = length(ray.dir);
        const float invRayLength = 1.f / rayLength;
        float3 unitRayDir = ray.dir * invRayLength;

        float2 result;
        bool isHit = intersectLinearSweptSphereHan19(ray.origin, unitRayDir, sphereA, sphereB, kUseSphereJoints, result);

        if (isHit)
        {
            t = result.x * invRayLength;
            bool isInRayExtent = (t >= ray.tMin && t <= ray.tMax);
            if (isInRayExtent)
            {
                // Back-face culling.
                float4 sphere = sphereA + result.y * (sphereB - sphereA);
                float3 hitPos = ray.origin + result.x * ray.dir;
                float3 normal = normalize(hitPos - sphere.xyz);

                if (dot(ray.dir, normal) < -1e-6f)
                {
                    // To avoid numerical issues, reprojecting from hitPos = ray.origin + t * ray.dir.
                    hitPos = sphere.xyz + sphere.w * normal;
                    attribs.barycentrics = curveHitPosToUV(hitPos, sphereA, sphereB);
                    return true;
                }
            }
        }

        return false;
    }
};

/** Helper for intersecting rays against SDF grids.
*/
struct SDFGridIntersector
{
    /** Intersects a ray with an SDF grid.
        \param[in] ray Ray in world-space.
        \param[in] instanceID Geometry instance ID.
        \param[in] primitiveID The primitive ID of the AABB that was hit.
        \param[out] hitData SDFGridHitData that contains hit point information.
        \param[out] t Intersection t.
        \return True if the ray intersects the SDF grid.
    */
    static bool intersect(const Ray ray, const GeometryInstanceID instanceID, const uint primitiveID, out SDFGridHitData hitData, out float t)
    {
        hitData ={};
        t = {};

        const GeometryInstanceData instance = gScene.getGeometryInstance(instanceID);
        SDFGrid sdfGrid;
        gScene.getSDFGrid(instanceID, sdfGrid);

        // Load Matrices.
        float4x4 worldMat = gScene.loadWorldMatrix(instance.globalMatrixID);
        float3x3 worldInvTransposeMat = gScene.loadInverseTransposeWorldMatrix(instance.globalMatrixID);

        // Convert ray origin and direction to local space.
        // Multiplying v.M to perform the intended (M^T).v
        float3 rayOrigLocal = mul(ray.origin - extractTranslate(worldMat), worldInvTransposeMat);
        float3 rayDirLocal = mul(ray.dir, worldInvTransposeMat);

        return sdfGrid.intersectSDF(rayOrigLocal, rayDirLocal, ray.tMin, ray.tMax, primitiveID, t, hitData);
    }

    /** Intersects a ray with an SDF grid, does not return information about the intersection.
        \param[in] ray Ray in world-space.
        \param[in] instanceID Geometry instance ID.
        \param[in] primitiveID The primitive ID of the AABB that was hit.
        \return True if the ray intersects the SDF grid.
    */
    static bool intersectAny(const Ray ray, const GeometryInstanceID instanceID, const uint primitiveID)
    {
        const GeometryInstanceData instance = gScene.getGeometryInstance(instanceID);
        SDFGrid sdfGrid;
        gScene.getSDFGrid(instanceID, sdfGrid);

        // Load Matrices.
        float4x4 worldMat = gScene.loadWorldMatrix(instance.globalMatrixID);
        float3x3 worldInvTransposeMat = gScene.loadInverseTransposeWorldMatrix(instance.globalMatrixID);

        // Convert ray origin and direction to local space.
        // Multiplying v.M to perform the intended (M^T).v
        float3 rayOrigLocal = mul(ray.origin - extractTranslate(worldMat), worldInvTransposeMat);
        float3 rayDirLocal = mul(ray.dir, worldInvTransposeMat);

#if SCENE_SDF_OPTIMIZE_VISIBILITY_RAYS
        return sdfGrid.intersectSDFAny(rayOrigLocal, rayDirLocal, ray.tMin, ray.tMax, primitiveID);
#else
        float dummyT;
        SDFGridHitData dummyHitData;
        return sdfGrid.intersectSDF(rayOrigLocal, rayDirLocal, ray.tMin, ray.tMax, primitiveID, dummyT, dummyHitData);
#endif
    }
}
