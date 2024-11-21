/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/

/** This is a test that all related API functions compile.
    The program is not actually executed, just compiled.
*/

// Dummy variables
RWStructuredBuffer<uint> result;                        // Dummy variable, not assigned
RaytracingAccelerationStructure gAccelerationStructure; // Dummy variable, not assigned

[numthreads(1, 1, 1)]
void testTraceRayInlineAPI(uint3 threadID: SV_DispatchThreadID)
{
    uint i = 0;

    // Instantiate ray query object.
    // Template parameter allows driver to generate a specialized implementation.
    RayQuery<RAY_FLAG_NONE> rayQuery;

    // Set up a trace. No work is done yet.
    RayDesc ray;
    ray.Origin = float3(0, 0, 0);
    ray.Direction = float3(1, 0, 0);
    ray.TMin = 0.f;
    ray.TMax = 1e30f;

    uint rayFlags = RAY_FLAG_NONE;
    uint instanceMask = 0xff;

    rayQuery.TraceRayInline(
        gAccelerationStructure,
        rayFlags, // OR'd with flags above
        instanceMask,
        ray
    );

    // Proceed() below is where behind-the-scenes traversal happens,
    // including the heaviest of any driver inlined code.
    while (rayQuery.Proceed())
    {
        // Query functions that are valid for all candidates.
        result[i++] = rayQuery.RayFlags();
        result[i++] = rayQuery.WorldRayOrigin().x;
        result[i++] = rayQuery.WorldRayDirection().x;
        result[i++] = rayQuery.RayTMin();
        result[i++] = rayQuery.CandidateInstanceIndex();
        result[i++] = rayQuery.CandidateInstanceID();
        result[i++] = rayQuery.CandidateInstanceContributionToHitGroupIndex();
        result[i++] = rayQuery.CandidateGeometryIndex();
        result[i++] = rayQuery.CandidatePrimitiveIndex();
        result[i++] = rayQuery.CandidateObjectRayOrigin().x;
        result[i++] = rayQuery.CandidateObjectRayDirection().x;
        result[i++] = rayQuery.CandidateObjectToWorld3x4()[0].x;
        result[i++] = rayQuery.CandidateObjectToWorld4x3()[0].x;
        result[i++] = rayQuery.CandidateWorldToObject3x4()[0].x;
        result[i++] = rayQuery.CandidateWorldToObject4x3()[0].x;

        // Proceed() returns true when there is a candidate for a hit that requires shader consideration.
        // Call RayQuery::CandidateType() to determine its type.
        if (rayQuery.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            // Query functions that are only valid for triangle hits.
            result[i++] = rayQuery.CandidateTriangleRayT();
            result[i++] = rayQuery.CandidateTriangleBarycentrics().x;
            result[i++] = rayQuery.CandidateTriangleFrontFace();

            // We hit a non-opaque triangle.
            // Call RayQuery::CommitNonOpaqueTriangleHit() to mark it as opaque.
            rayQuery.CommitNonOpaqueTriangleHit();

            // Abort the traversal. We don't care about other hits.
            rayQuery.Abort();
        }
        else if (rayQuery.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE)
        {
            // Query functions that are only valid for procedural candidates.
            result[i++] = rayQuery.CandidateProceduralPrimitiveNonOpaque();

            // We hit a procedural primitive.
            // Call RayQuery::CommitProceduralPrimitiveHit(float tHit) for the computed intersection.
            rayQuery.CommitProceduralPrimitiveHit(100.f);
        }
    }

    // We have looped until Proceed() returned false, which means there are no more candidates to evaluate.
    // Call RayQuery::CommittedStatus() to examine the result of the traversal.
    if (rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        // Query functions that are available for all committed hits.
        result[i++] = rayQuery.CommittedRayT();
        result[i++] = rayQuery.CommittedInstanceIndex();
        result[i++] = rayQuery.CommittedInstanceID();
        result[i++] = rayQuery.CommittedInstanceContributionToHitGroupIndex();
        result[i++] = rayQuery.CommittedGeometryIndex();
        result[i++] = rayQuery.CommittedPrimitiveIndex();
        result[i++] = rayQuery.CommittedObjectRayOrigin().x;
        result[i++] = rayQuery.CommittedObjectRayDirection().x;
        result[i++] = rayQuery.CommittedObjectToWorld3x4()[0].x;
        result[i++] = rayQuery.CommittedObjectToWorld4x3()[0].x;
        result[i++] = rayQuery.CommittedWorldToObject3x4()[0].x;
        result[i++] = rayQuery.CommittedWorldToObject4x3()[0].x;

        // Query functions that are only valid for triangle hits.
        result[i++] = rayQuery.CommittedTriangleBarycentrics().x;
        result[i++] = rayQuery.CommittedTriangleFrontFace();
    }
    else if (rayQuery.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT)
    {
        // We hit a procedural primitive.
    }
    else if (rayQuery.CommittedStatus() == COMMITTED_NOTHING)
    {
        // We missed. There is no committed hit.
    }
}
