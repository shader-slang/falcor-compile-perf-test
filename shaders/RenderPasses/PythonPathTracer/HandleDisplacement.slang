/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
import Scene.Displacement.DisplacementMapping;
import Scene.Shading;

DisplacementData getDisplacementData(uint materialID)
{
    const MaterialSystem msys = gScene.materials;
    DisplacementData displacementData;
    msys.loadDisplacementData(materialID, displacementData);
    displacementData.scale = 0.1; // TODO: Hardcoded
    displacementData.bias = -1;
    return displacementData;
}

struct DisplacementContext
{
    bool valid;
    StaticVertexData vertices[3];
    uint materialID;

    float3 curPos;
    float3 curBary;
    float curHeight;
    float hTop;
    float hBot;

    __init(ShadingData sd, StaticVertexData vertices_[3])
    {
        valid = gScene.materials.isDisplaced(sd.materialID);
        if (valid)
        {
            vertices = vertices_;
            materialID = sd.materialID;
            grow();

            curPos = sd.posW;
            curBary = uvToBary(sd.uv);
            DisplacementData data = getDisplacementData(sd.materialID);
            hTop = data.mapValue(1.0f);
            hBot = data.mapValue(0.0f);
            curHeight = data.readValue(sd.uv * data.size, 0.0f);
        }
    }

    float3 uvToBary(float2 uv)
    {
        float2 A = vertices[0].texCrd - vertices[2].texCrd;
        float2 B = vertices[1].texCrd - vertices[2].texCrd;
        float2x2 M = float2x2(B.y, -B.x, -A.y, A.x) / (A.x * B.y - A.y * B.x);
        float2 ab = mul(M, uv - vertices[2].texCrd);
        return float3(ab[0], ab[1], 1.0f - ab[0] - ab[1]);
    }

    [mutating]
    void grow()
    {
        float3 posC = (vertices[0].position + vertices[1].position + vertices[2].position) / 3.0f;
        float2 uvC = (vertices[0].texCrd + vertices[1].texCrd + vertices[2].texCrd) / 3.0f;

        float S = 10.0f;
        vertices[0].position += (vertices[0].position - posC) * S;
        vertices[1].position += (vertices[1].position - posC) * S;
        vertices[2].position += (vertices[2].position - posC) * S;
        vertices[0].texCrd += (vertices[0].texCrd - uvC) * S;
        vertices[1].texCrd += (vertices[1].texCrd - uvC) * S;
        vertices[2].texCrd += (vertices[2].texCrd - uvC) * S;
    }

    bool displace(float3 pos, float3 dir, out DisplacementIntersection result)
    {
        float4x4 tform = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
        DisplacementData data = getDisplacementData(materialID);
        return intersectDisplacedTriangle(Ray(pos, dir), vertices, tform, data, result);
    }

    void applyNormal(inout ShadingData sd)
    {
        float4x4 tform = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
        float3x3 invTform = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
        DisplacementData data = getDisplacementData(materialID);
        float3 N = computeDisplacedTriangleNormal(curBary, vertices, tform, invTform, data);

        sd.frame = ShadingFrame(N, sd.tangentW);
    }

    [mutating]
    bool primaryHit(inout ShadingData sd, float3 dir)
    {
        if (!valid)
            return false;
        DisplacementIntersection hit;
        if (!displace(sd.posW, dir, hit))
            return false;
        curPos += dir * hit.t;
        curBary = hit.getBarycentricWeights();
        curHeight = hit.displacement;
        applyNormal(sd);
        sd.uv = curBary[0] * vertices[0].texCrd + curBary[1] * vertices[1].texCrd + curBary[2] * vertices[2].texCrd;
        return true;
    }

    bool secondaryHit(float3 dir)
    {
        if (!valid)
            return false;
        float exitT = 10.0f; // TODO?
        DisplacementIntersection hit;
        if (!displace(curPos + exitT * dir, -dir, hit))
            return false;
        return (hit.t < exitT - 1e-3f);
    }
};

DisplacementContext makeDisplacementContext(ShadingData sd, StaticVertexData vertices[3])
{
    return DisplacementContext(sd, vertices);
}

DisplacementContext makeDisplacementContext(ShadingData sd, const TriangleHit triangleHit)
{
    StaticVertexData vertices[3];
    VertexData v = gScene.getVertexData(triangleHit, vertices);

    return makeDisplacementContext(sd, vertices);
}
