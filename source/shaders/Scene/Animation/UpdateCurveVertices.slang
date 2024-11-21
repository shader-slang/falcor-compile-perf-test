/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import Scene.SceneTypes;

#ifndef CURVE_KEYFRAME_COUNT
// CURVE_KEYFRAME_COUNT must be defined in order to use this module.
#error "CURVE_KEYFRAME_COUNT not defined!"
#endif

/** Compute pass for updating animated vertices.

    The dispatch size is one thread per dynamic vertex.
*/

struct CurvePerKeyframe
{
    StructuredBuffer<DynamicCurveVertexData> vertexData;
};

struct CurveVertexUpdater
{
    uint2 keyframeIndices;
    float t;
    bool copyPrev;

    uint dimX;
    uint vertexCount;

    // Curve vertex caches at all keyframes
#if CURVE_KEYFRAME_COUNT > 0
    CurvePerKeyframe curvePerKeyframe[CURVE_KEYFRAME_COUNT];
#else
    CurvePerKeyframe curvePerKeyframe[1];
#endif

    // Output
    RWStructuredBuffer<DynamicCurveVertexData> prevCurveVertices;
    RWStructuredBuffer<StaticCurveVertexData> curveVertices;

    // Accessors
    DynamicCurveVertexData interpolateCurveVertex(uint vertexID)
    {
        DynamicCurveVertexData v0, v1;
        v0 = curvePerKeyframe[keyframeIndices.x].vertexData[vertexID];
        v1 = curvePerKeyframe[keyframeIndices.y].vertexData[vertexID];

        DynamicCurveVertexData result;
        result.position = lerp(v0.position, v1.position, t);
        return result;
    }

    void updateVertex(uint3 dispatchThreadID)
    {
        uint vertexID = dispatchThreadID.y * dimX + dispatchThreadID.x;
        if (vertexID >= vertexCount) return;

        if (copyPrev)
        {
            prevCurveVertices[vertexID].position = curveVertices[vertexID].position;
        }
        else
        {
            float3 curPos = interpolateCurveVertex(vertexID).position;
            float3 prevPos = curveVertices[vertexID].position;

            prevCurveVertices[vertexID].position = prevPos;
            curveVertices[vertexID].position = curPos;
        }
    }
};

ParameterBlock<CurveVertexUpdater> gCurveVertexUpdater;

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    gCurveVertexUpdater.updateVertex(dispatchThreadID);
}
