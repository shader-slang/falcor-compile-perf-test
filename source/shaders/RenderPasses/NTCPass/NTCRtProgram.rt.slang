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
import Internal.Utils.Neural.TIN.TinEncoding;

import Scene.Raytracing;

import Utils.Debug.PixelDebug;
import Utils.Geometry.IntersectionHelpers;
import Utils.Sampling.TinyUniformSampleGenerator;

import GeneratedMLP; // This module is generated, not loaded from file.

#ifndef NTC_SW_TRILINEAR
#error Expected NTC_SW_TRILINEAR to be defined
#endif

Buffer<uint> gFeatureGrids;
Texture2D<PackedHitInfo> gVBuffer;
Texture2D<float4> gViewW;
Texture2D<float4> gTextureGrads;

StructuredBuffer<NetworkParamStorageFP16> gParams;
// These buffers alias the same resource as `gParams` above but uses different struct types.
StructuredBuffer<Tin::MMAMatStore> gTinWeights;
StructuredBuffer<uint> gTinBiases;
StructuredBuffer<float16_t> gTinParamsNVCoopVec;

Texture2D<float> gSTBN;
RWTexture2D<float4> gOutput;

cbuffer PerFrameCB
{
    float minLod;
    float maxLod;

    uint sampleIndex;
    uint uvJitterMode;
    uint lodJitterMode;
    uint hiresGridMipOffset[8];
    uint loresGridMipOffset[8];
    uint minNeuralLod;
    uint maxNeuralLod;
    uint useSTBN;
    uint paramsByteOffset;
}

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

[shader("raygeneration")]
void rayGen()
{
    uint2 launchIndex = DispatchRaysIndex().xy;
    const HitInfo hit = HitInfo(gVBuffer[launchIndex]);
    TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(launchIndex.xy, sampleIndex);

    // Initialize output color.
    float4 color = float4(0, 0.3, 0.5, 1.f);

    bool shade = hit.isValid() && hit.getType() == HitType::Triangle;

    printSetPixel(launchIndex);

    // Prepare the shading data.
    ShadingData sd;
    const TriangleHit triangleHit = hit.getTriangleHit();
    const VertexData v = gScene.getVertexData(triangleHit);

    uint materialID = gScene.getMaterialID(triangleHit.instanceID);
    let matsam = ExplicitLodTextureSampler(0.f);
    float3 rayDirW = -gViewW[launchIndex].xyz;
    sd = gScene.materials.prepareShadingData(v, materialID, -rayDirW, matsam);

    float4 texGrads = gTextureGrads[launchIndex];

    float3 albedo = 0, normal = 0;
    float ao = 0, roughness = 0, metallic = 0, specular = 0;

    // STBN has only 256 possible values 0-255, so add a random value to cover all possible ones.
    float noise = gSTBN.Load(int3((launchIndex.xy) % 128, 0)) * 255.0 / 256.0 + sampleNext1D(sg) / 256.0;

    let uvJitter = getJitter(sg, uvJitterMode);
    let lodJitter = getJitter(sg, lodJitterMode)[0];

    let lod = computeLod(uint2(NTC_GRID_WIDTH, NTC_GRID_HEIGHT), texGrads, minLod, maxLod, lodJitter);

#if NTC_SW_TRILINEAR == 0
    float w = 1;
    float2 uv = v.texC;
    let int_lod = int(lod);
    let neuralLod = lod2NeuralLod(lod, minNeuralLod, maxNeuralLod);
#else
    for (int l = 0; l < 2; l++)
    {
        let int_lod = l + int(lod);
        let neuralLod = lod2NeuralLod(int_lod, minNeuralLod, maxNeuralLod);

        let uv_scale = float2(NTC_GRID_WIDTH >> int_lod, NTC_GRID_HEIGHT >> int_lod);
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

    // Note that HPackedArray requires element count to be a multiple of two.
    Tin::HPackedArray<MLP_IP_CHANNELS_PADDED> ip;
    ip.clear();

#if NTC_SEL_FEATURE < 8
    Tin::FeatureGrid<NTC_GRID_WIDTH, NTC_GRID_HEIGHT, NTC_FEATURE_CHANNELS_HRES, NTC_FEATURE_BITS_HRES, NTC_FEATURE_ALLCORNER_HRES>
        featureGridHres;

    uv = roundToPixel(uv, uvJitter, float2(NTC_GRID_WIDTH >> int_lod, NTC_GRID_HEIGHT >> int_lod));

    uint bufferOffset = hiresGridMipOffset[neuralLod];
    uint slotOffset = NTC_FEATURE_CHANNELS_LRES / 2;

    featureGridHres.sample(uv, NTC_START_GRID_SCALE << (2 * neuralLod), bufferOffset, slotOffset, gFeatureGrids, ip, NTC_SEL_FEATURE);

#if NTC_NUM_FEATURE_GRIDS > 1
    bufferOffset = loresGridMipOffset[neuralLod];
    slotOffset = 0;

    Tin::FeatureGrid<NTC_GRID_WIDTH, NTC_GRID_HEIGHT, NTC_FEATURE_CHANNELS_LRES, NTC_FEATURE_BITS_LRES, false> featureGridLres;
    featureGridLres.sample(uv, NTC_START_GRID_SCALE << (1 + 2 * neuralLod), bufferOffset, slotOffset, gFeatureGrids, ip, NTC_SEL_FEATURE);
#endif
#endif

#if NTC_ENCODING == 0
    uint encoding_offset; // Offset in scalars.
    if (NTC_FEATURE_ALLCORNER_HRES)
    {
        encoding_offset = NTC_FEATURE_CHANNELS_HRES * 4 + NTC_FEATURE_CHANNELS_LRES;
    }
    else
    {
        encoding_offset = NTC_FEATURE_CHANNELS_HRES + NTC_FEATURE_CHANNELS_LRES;
    }

    let triEncoding = Tin::HTriEncoding<NTC_START_GRID_SCALE << (NTC_NUM_FEATURE_GRIDS - 1)>(NTC_GRID_WIDTH, NTC_GRID_HEIGHT);
    encoding_offset = triEncoding.encode(ip, uv, encoding_offset);
#endif

    // Add normalized LOD to MLP input (note: second half of the 32b register is currently unused)
    // Note that set_packed_item assumes offset in half2 pairs => divide scalar encoding_offset by two.
    ip.set_packed_item(encoding_offset / 2, half2::pack(half2(normalizeLod(int_lod, maxLod), 0)));

    // Evaluate network.
    // The `NTCMLP` type is defined in the imported generated Slang module.
    float16_t ipf[MLP_IP_CHANNELS] = {};
    [unroll]
    for (int i = 0; i < MLP_IP_CHANNELS; i++)
    {
        ipf[i] = ip[i];
    }
    float16_t op[MLP_OP_CHANNELS] = {};

    NTCMLP mlp;
    mlp.eval(gParams, gTinWeights, gTinBiases, gTinParamsNVCoopVec, ipf, op, paramsByteOffset);

#if NTC_DIFFUSE_CHANNEL >= 0
    albedo += w * float3(op[NTC_DIFFUSE_CHANNEL], op[NTC_DIFFUSE_CHANNEL + 1], op[NTC_DIFFUSE_CHANNEL + 2]);
#endif
#if NTC_NORMAL_CHANNEL >= 0
    normal += w * float3(op[NTC_NORMAL_CHANNEL], op[NTC_NORMAL_CHANNEL + 1], op[NTC_NORMAL_CHANNEL + 2]);
#else
                normal += w * float3(0, 0, 1);
#endif

#if NTC_AO_CHANNEL >= 0
    ao += w * op[NTC_AO_CHANNEL];
#endif
#if NTC_METALLIC_CHANNEL >= 0
    metallic += w * op[NTC_METALLIC_CHANNEL];
#endif
#if NTC_ROUGHNESS_CHANNEL >= 0
    roughness += w * op[NTC_ROUGHNESS_CHANNEL];
#endif
#if NTC_SPECULAR_CHANNEL >= 0
    specular += w * op[NTC_SPECULAR_CHANNEL];
#else
                specular += w;
#endif

#if NTC_SW_TRILINEAR
}
}
}
#endif

float4 colorHit = shadeSample(pow(albedo, 2.2), normal, ao, roughness, metallic, specular, sd);

if (shade)
{
    color = colorHit;
}

gOutput[launchIndex.xy] = color;
}
