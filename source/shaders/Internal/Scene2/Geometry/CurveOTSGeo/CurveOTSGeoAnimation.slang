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
import Internal.Scene2.Geometry.GeoBase;
import Internal.Scene2.RWSceneGeo;
import Internal.Scene2.Geometry.CurveOTSGeo.CurveOTSGeoSharedTypes;
import Utils.SlangUtils;
import Utils.Math.Quaternion;
import Internal.Scene2.Geometry.GeoRWAccessors;
#include "CurveOTSGeoAnimationTypes.slangh"

struct CurveOTSGeoAnimation
{
    StructuredBuffer<CurveOTSVertexAnimation> curveKeyframes[ArrayMax<1, CURVE_OTS_COUNT>.value];
    StructuredBuffer<CurveOTSVertexType> curveVertexTypes[ArrayMax<1, CURVE_OTS_COUNT>.value];
    StructuredBuffer<CurveAnimationInfo> curveAnimationInfo;

    float3 interpolateCurvePosition(uint curveAnimationIndex, uint animVertexIndex, CurveAnimationInfo curveAnimationInfo)
    {
        float3 v0 = curveKeyframes[curveAnimationIndex][curveAnimationInfo.keyframeVertexBases[0] + animVertexIndex].position;
        float3 v1 = curveKeyframes[curveAnimationIndex][curveAnimationInfo.keyframeVertexBases[1] + animVertexIndex].position;
        return lerp(v0, v1, curveAnimationInfo.t);
    }

    // staticVertexOffset - index within the curve (add to CurveAnimationInfo values to get static/anim vertex)
    // curveAnimationIndex - index of the curve (points into curveAnimationInfo)
    void updateVertex(uint staticVertexOffset, uint curveAnimationIndex)
    {
        let curveAnimationInfo = curveAnimationInfo[curveAnimationIndex];
        if (staticVertexOffset >= curveAnimationInfo.staticVertexCount)
            return;

        // There is one animated vertex for every 4 static vertices
        uint animVertexIndex = staticVertexOffset / 4;
        uint staticVertexIndex = curveAnimationInfo.staticVertexBase + staticVertexOffset;

        float3 curCurvePos = interpolateCurvePosition(curveAnimationIndex, animVertexIndex, curveAnimationInfo);
        CurveOTSVertexType vertexType = curveVertexTypes[curveAnimationIndex][animVertexIndex];

        // Get position of the neighboring vertex, next unless at the end of a strand
        const int animNeighborOffset = (vertexType != CurveOTSVertexType::StrandEnd) ? 1 : -1;
        const float3 curCurvePosNeighbor = interpolateCurvePosition(curveAnimationIndex, animVertexIndex + 1, curveAnimationInfo);
        const float3 curCurveTangent = normalize(curCurvePosNeighbor - curCurvePos);

        CurveOTSStaticVertex meshVertex = gRWScene2Geo.curveOTSGeoData.vertices[staticVertexIndex];
        float3 prevMeshTangent = meshVertex.curveTangent;
        float4 quat = from_to_rotation(prevMeshTangent, curCurveTangent);
        float3 curNormal = rotate_vector(meshVertex.fromCenter, quat);

        meshVertex.curveTangent = curCurveTangent;
        meshVertex.fromCenter = curNormal;
        meshVertex.position = curCurvePos + meshVertex.radius * meshVertex.fromCenter;

        if (any(isnan(meshVertex.position)) || any(isinf(meshVertex.position)))
        {
            meshVertex.position = curCurvePos;
        }

        gRWScene2Geo.curveOTSGeoData.vertices[staticVertexIndex] = meshVertex;
    }
};

ParameterBlock<CurveOTSGeoAnimation> gCurveOTSGeoAnimation;

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    gCurveOTSGeoAnimation.updateVertex(dispatchThreadID.x, dispatchThreadID.y);
}
