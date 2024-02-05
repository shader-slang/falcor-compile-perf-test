/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
import Scene.SceneTypes;
import Utils.Math.Quaternion;
import Utils.Sampling.SampleGenerator;
import SharedTypes;

#include "Utils/Math/MathConstants.slangh"

#ifndef CURVE_KEYFRAME_COUNT
// CURVE_KEYFRAME_COUNT must be defined in order to use this module.
#error "CURVE_KEYFRAME_COUNT not defined!"
#endif

struct CurvePerKeyframe
{
    StructuredBuffer<DynamicCurveVertexData> vertexData;
};

struct CurveDisjointOTSVertexUpdater
{
    uint2 keyframeIndices;
    float t;
    uint seed;
    bool copyPrev;

    // Curve vertex caches at all keyframes
#if CURVE_KEYFRAME_COUNT > 0
    CurvePerKeyframe curvePerKeyframe[CURVE_KEYFRAME_COUNT];
#else
    CurvePerKeyframe curvePerKeyframe[1];
#endif

    ByteAddressBuffer curveIndexData;
    StructuredBuffer<PerMeshMetadata> perMeshData;
    StructuredBuffer<PerCurveMetadata> perCurveData;

    // Output
    RWStructuredBuffer<PackedStaticVertexData> sceneVertexData;
    RWStructuredBuffer<PrevVertexData> prevVertexData;
    RWStructuredBuffer<StaticCurveVertexData> curveVertexData;

    float3 perp_stark(float3 u)
    {
        // TODO: Validate this and look at numerical precision etc. Are there better ways to do it?
        float3 a = abs(u);
        uint uyx = (a.x - a.y) < 0 ? 1 : 0;
        uint uzx = (a.x - a.z) < 0 ? 1 : 0;
        uint uzy = (a.y - a.z) < 0 ? 1 : 0;
        uint xm = uyx & uzx;
        uint ym = (1 ^ xm) & uzy;
        uint zm = 1 ^ (xm | ym); // 1 ^ (xm & ym)
        float3 v = normalize(cross(u, float3(xm, ym, zm)));
        return v;
    }

    void buildFrame(in float3 n, out float3 t, out float3 b)
    {
        t = perp_stark(n);
        b = cross(n, t);
    }

    void getNormalAndTangent<S : ISampleGenerator>(
        uint threadID,
        float3 curveVertexPos,
        float3 curveVertexPosNext,
        inout S sg,
        out float3 normal,
        out float3 tangent
    )
    {
        float3 fwd = normalize(curveVertexPosNext - curveVertexPos);
        float3 s, t;
        buildFrame(fwd, s, t);

        float phi = (threadID % 4) / 4.f;
        phi += sampleNext1D(sg);
        phi *= M_PI * 2.f;

        normal = cos(phi) * s + sin(phi) * t;
        tangent = fwd;
    }

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

    void updateVertex(uint threadID, uint curveID)
    {
        const uint meshVertexIndex = threadID;

        // Curve and mesh metadata is placed at the same place in respective buffers.
        const uint meshID = curveID;
        PerCurveMetadata curveMeta = perCurveData[curveID];
        PerMeshMetadata meshMeta = perMeshData[meshID];

        if (meshVertexIndex >= meshMeta.vertexCount)
            return;

        const uint curveVertexIndex = curveIndexData.Load(4 * (curveMeta.indexOffset + threadID / 8));
        const uint globalCurveVertexIndex = curveMeta.vertexOffset + curveVertexIndex;

        const uint globalMeshVertexIndex = meshMeta.sceneVbOffset + meshVertexIndex;
        const uint globalPrevVertexIndex = meshMeta.prevVbOffset + meshVertexIndex;

        if (copyPrev)
        {
            StaticVertexData meshVertex = sceneVertexData[globalMeshVertexIndex].unpack();

            prevVertexData[globalPrevVertexIndex].position = meshVertex.position;

            // Copy mesh data into curve vertex data buffers.
            StaticCurveVertexData curveVertex = {};
            curveVertex.position = meshVertex.position - meshVertex.curveRadius * meshVertex.normal;
            curveVertex.radius = meshVertex.curveRadius;
            curveVertex.texCrd = meshVertex.texCrd;

            if (threadID % 8 == 0)
            {
                curveVertexData[globalCurveVertexIndex] = curveVertex;
            }
            else if (threadID % 8 == 4)
            {
                curveVertexData[globalCurveVertexIndex + 1] = curveVertex;
            }
        }
        else
        {
            StaticCurveVertexData prevCurveVertex = curveVertexData[globalCurveVertexIndex];
            StaticCurveVertexData prevCurveVertexNext = curveVertexData[globalCurveVertexIndex + 1];

            // Update prevVertexData.
            {
                SampleGenerator sg = SampleGenerator(seed - 1 + threadID / 8);
                float3 prevNormal, prevTangent;
                getNormalAndTangent(threadID, prevCurveVertex.position, prevCurveVertexNext.position, sg, prevNormal, prevTangent);

                if (meshVertexIndex % 8 < 4)
                {
                    prevVertexData[globalPrevVertexIndex].position = prevCurveVertex.position + prevCurveVertex.radius * prevNormal;
                }
                else
                {
                    prevVertexData[globalPrevVertexIndex].position = prevCurveVertexNext.position + prevCurveVertexNext.radius * prevNormal;
                }
            }

            StaticCurveVertexData curCurveVertex = prevCurveVertex;
            StaticCurveVertexData curCurveVertexNext = prevCurveVertexNext;
            curCurveVertex.position = interpolateCurveVertex(globalCurveVertexIndex).position;
            curCurveVertexNext.position = interpolateCurveVertex(globalCurveVertexIndex + 1).position;

            SampleGenerator sg = SampleGenerator(seed + threadID / 8);
            float3 curNormal, curTangent;
            getNormalAndTangent(threadID, curCurveVertex.position, curCurveVertexNext.position, sg, curNormal, curTangent);

            StaticVertexData meshVertex = {};
            meshVertex.normal = curNormal;
            meshVertex.tangent = float4(curTangent, 1.f);

            if (meshVertexIndex % 8 < 4)
            {
                meshVertex.texCrd = curCurveVertex.texCrd;
                meshVertex.curveRadius = curCurveVertex.radius;
                meshVertex.position = curCurveVertex.position + meshVertex.curveRadius * meshVertex.normal;

                if (any(isnan(meshVertex.position)) || any(isinf(meshVertex.position)))
                {
                    meshVertex.position = curCurveVertex.position;
                }
            }
            else
            {
                meshVertex.texCrd = curCurveVertexNext.texCrd;
                meshVertex.curveRadius = curCurveVertexNext.radius;
                meshVertex.position = curCurveVertexNext.position + meshVertex.curveRadius * meshVertex.normal;

                if (any(isnan(meshVertex.position)) || any(isinf(meshVertex.position)))
                {
                    meshVertex.position = curCurveVertexNext.position;
                }
            }

            sceneVertexData[globalMeshVertexIndex].pack(meshVertex);

            curveVertexData[globalCurveVertexIndex] = curCurveVertex;
            curveVertexData[globalCurveVertexIndex + 1] = curCurveVertexNext;
        }
    }
};

ParameterBlock<CurveDisjointOTSVertexUpdater> gCurveDisjointOTSVertexUpdater;

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    gCurveDisjointOTSVertexUpdater.updateVertex(dispatchThreadID.x, dispatchThreadID.y);
}
