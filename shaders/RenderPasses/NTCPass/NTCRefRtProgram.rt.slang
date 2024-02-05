/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
import RenderPasses.NTCPass.BRDF;
import RenderPasses.NTCPass.NTCCommon;

import Scene.Raytracing;
import Rendering.Lights.LightHelpers;

import Utils.Debug.PixelDebug;
import Utils.Geometry.IntersectionHelpers;
import Utils.Sampling.TinyUniformSampleGenerator;

#ifndef NTC_SW_TRILINEAR
#error Expected NTC_SW_TRILINEAR to be defined
#endif

SamplerState gSampler;
Texture2D<float4> gRefTex[NTC_REF_NUM_TEXTURES];

Texture2D<PackedHitInfo> gVBuffer;
Texture2D<float4> gViewW;
Texture2D<float4> gTextureGrads;

cbuffer PerFrameCB
{
    float minLod;
    float maxLod;

    uint sampleIndex;
    uint uvJitterMode;
    uint lodJitterMode;
};

struct PrimaryRayData
{
    float hitT;
    uint triangleIndex;
    GeometryInstanceID instanceID;
    BuiltInTriangleIntersectionAttributes attribs;
};

struct ShadowRayData
{
    bool hit;
};

struct RayData
{
    float t;
    float3 color;
};

[shader("miss")]
void shadowMiss(inout ShadowRayData hitData)
{
    hitData.hit = false;
}

[shader("anyhit")]
void shadowAnyHit(inout ShadowRayData hitData, BuiltInTriangleIntersectionAttributes attribs)
{
    hitData.hit = true;
}

[shader("miss")]
void primaryMiss(inout PrimaryRayData hitData)
{
    hitData.hitT = -1.f;
}

[shader("closesthit")]
void primaryClosestHit(inout PrimaryRayData hitData, BuiltInTriangleIntersectionAttributes attribs)
{
    hitData.triangleIndex = PrimitiveIndex();
    hitData.hitT = RayTCurrent();
    hitData.instanceID = getGeometryInstanceID();
    hitData.attribs = attribs;
}

[shader("anyhit")]
void primaryAnyHit(inout PrimaryRayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    // Alpha test for non-opaque geometry.
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    uint materialID = gScene.getMaterialID(instanceID);
    if (gScene.materials.alphaTest(v, materialID, 0.f))
    {
        IgnoreHit();
    }
}

//
// Entry point for ray generation shader.
//

RWTexture2D<float4> gOutput;

[shader("raygeneration")]
void rayGen()
{
    uint2 launchIndex = DispatchRaysIndex().xy;
    const HitInfo hit = HitInfo(gVBuffer[launchIndex]);
    TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(launchIndex.xy, sampleIndex);

    // Initialize output color.
    float4 color = float4(0, 0.3, 0.5, 1.f);

    bool shade = hit.isValid() && hit.getType() == HitType::Triangle;
    if (shade)
    {
        printSetPixel(launchIndex);

        let uvJitter = getJitter(sg, uvJitterMode);
        let lodJitter = getJitter(sg, lodJitterMode)[0];

        // Prepare the shading data.
        const TriangleHit triangleHit = hit.getTriangleHit();
        const VertexData v = gScene.getVertexData(triangleHit);
        uint materialID = gScene.getMaterialID(triangleHit.instanceID);

        uint2 refDim;
        float4 texGrads = gTextureGrads[launchIndex];
        gRefTex[0].GetDimensions(refDim.x, refDim.y);
        let lod = computeLod(refDim, texGrads, minLod, maxLod, lodJitter);

#if NTC_SW_TRILINEAR
        let matsam = ExplicitLodTextureSampler(int(lod));
#else
        let matsam = ExplicitLodTextureSampler(int(lod));
        let texsam = matsam;
#endif
        let rayDirW = -gViewW[launchIndex].xyz;
        let sd = gScene.materials.prepareShadingData(v, materialID, -rayDirW, matsam);

        float4 data;
        float3 albedo = 0, normal = 0;
        float ao = 0, roughness = 0, metallic = 0, specular = 0;

#if NTC_SW_TRILINEAR == 0
        float w = 1.f;
        float2 uv = v.texC + uvJitter / float2(NTC_REF_WIDTH, NTC_REF_HEIGHT);
#else
        for (int l = 0; l < 2; l++)
        {
            let int_lod = l + int(lod);
            let texsam = ExplicitLodTextureSampler(int_lod);

            let uv_scale = float2(NTC_REF_WIDTH >> int_lod, NTC_REF_HEIGHT >> int_lod);
            float2 uv = v.texC * uv_scale - 0.5;
            let corner_uv = floor(uv);

            let frac_u = uv[0] - corner_uv[0];
            let frac_v = uv[1] - corner_uv[1];
            let frac_lod = frac(lod);

            let a = l == 0 ? (1 - frac_lod) : frac_lod;
            for (float u = 0; u < 2; u++)
            {
                let b = u == 0 ? (1 - frac_u) : frac_u;
                for (float v = 0; v < 2; v++)
                {
                    let c = v == 0 ? (1 - frac_v) : frac_v;
                    let w = a * b * c;

                    uv = corner_uv + float2(u, v);
                    uv = uv / uv_scale;
#endif

#if NTC_REF_DIFFUSE_SLOT >= 0
        data = texsam.sampleTexture(gRefTex[NTC_REF_DIFFUSE_SLOT], gSampler, uv);
        albedo += w * data.xyz;
#endif

#if NTC_REF_NORMAL_SLOT >= 0
        data = texsam.sampleTexture(gRefTex[NTC_REF_NORMAL_SLOT], gSampler, uv);
        normal += w * data.xyz;
#else
                    normal += w * float3(0, 0, 1);
#endif

#if NTC_REF_AO_SLOT >= 0
        data = texsam.sampleTexture(gRefTex[NTC_REF_AO_SLOT], gSampler, uv);
        ao += w * data.x;
#endif

#if NTC_REF_ROUGHNESS_SLOT >= 0
        data = texsam.sampleTexture(gRefTex[NTC_REF_ROUGHNESS_SLOT], gSampler, uv);
        roughness += w * data.x;
#endif

#if NTC_REF_METALLIC_SLOT >= 0
        data = texsam.sampleTexture(gRefTex[NTC_REF_METALLIC_SLOT], gSampler, uv);
        metallic += w * data.x;
#endif

#if NTC_REF_SPECULAR_SLOT >= 0
        data = texsam.sampleTexture(gRefTex[NTC_REF_SPECULAR_SLOT], gSampler, uv);
        specular += w * data.x;
#else
                    specular += w;
#endif

#if NTC_SW_TRILINEAR
    }
}
}
#endif

color = shadeSample(albedo, normal, ao, roughness, metallic, specular, sd);
}

gOutput[launchIndex.xy] = color;
}
