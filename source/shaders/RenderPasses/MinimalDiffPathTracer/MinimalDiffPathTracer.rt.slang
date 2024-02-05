/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Scene.Raytracing;
import Scene.Intersection;
import Scene.RaytracingInline;
import Utils.Math.MathHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.SampleGenerator;
import Rendering.Lights.LightHelpers;
import BsdfGGX;
import MaterialParamsIO;

cbuffer CB
{
    uint gFrameCount;    // Frame count since scene was loaded.
    uint gPRNGDimension; // First available PRNG dimension.
}

// Inputs
Texture2D<PackedHitInfo> gVBuffer;
Texture2D<float4> gViewW; // Optional

// Outputs
RWTexture2D<float4> gOutputColor;
RWTexture2D<float4> gOutputFwdDColor;

// Static configuration based on defines set from the host.
#define is_valid(name) (is_valid_##name != 0)
static const uint kMaxBounces = MAX_BOUNCES;
static const bool kComputeDirect = COMPUTE_DIRECT;
static const bool kUseImportanceSampling = USE_IMPORTANCE_SAMPLING;
static const bool kUseAnalyticLights = USE_ANALYTIC_LIGHTS;
static const bool kUseEmissiveLights = USE_EMISSIVE_LIGHTS;
static const bool kUseEnvLight = USE_ENV_LIGHT;
static const bool kUseEnvBackground = USE_ENV_BACKGROUND;
static const float3 kDefaultBackgroundColor = float3(0, 0, 0);
static const float kRayTMax = FLT_MAX;

struct PathData : IDifferentiable
{
    float3 radiance;
    float3 thp;
    uint length;
    bool terminated;
    bool isHit;
    uint2 pixel;

    [Differentiable]
    __init(uint2 pixel)
    {
        this.radiance = float3(0.f);
        this.thp = float3(1.f);
        this.length = 0;
        this.terminated = false;
        this.isHit = false;
        this.pixel = pixel;
    }
}

bool traceRayInline(const Ray ray, out HitInfo hit)
{
    SceneRayQuery<0> sceneRayQuery;
    float hitT = 0.f;
    return sceneRayQuery.traceRay(ray, hit, hitT, RAY_FLAG_NONE, 0xff);
}

ShadingData loadShadingData(const HitInfo hit, const float3 rayDir, const ITextureSampler lod)
{
    VertexData v = {};
    uint materialID = {};

    if (hit.getType() == HitType::Triangle)
    {
        const TriangleHit triangleHit = hit.getTriangleHit();
        v = gScene.getVertexData(triangleHit);
        materialID = gScene.getMaterialID(triangleHit.instanceID);
    }

    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -rayDir, lod);
    return sd;
}

[Differentiable]
[PreferRecompute]
bool generateScatterRay(const ShadingData sd, const BSDFParameters bsdfParams, float3 rayOrigin, inout SampleGenerator sg, inout PathData pathData, inout Ray ray)
{
    BSDFScatterSample result;
    bool valid = bsdfGGXSample(sd, bsdfParams, sg, result);

    if (valid)
    {
        ray.origin = detach(rayOrigin);
        ray.dir = detach(result.wo);
        pathData.thp *= result.weight;
    }

    return valid;
}

[Differentiable]
[PreferRecompute]
void handleHit(const HitInfo hit, inout SampleGenerator sg, inout PathData pathData, inout Ray ray)
{
    let lod = ExplicitLodTextureSampler(0.f);

    // Load shading data.
    ShadingData sd = no_diff loadShadingData(hit, ray.dir, lod);

    // Create material instance.
    let mi = gScene.materials.getMaterialInstance(sd, lod);

    // Add emitted light.
    if (kUseEmissiveLights && (kComputeDirect || pathData.length > 0))
    {
        pathData.radiance += pathData.thp * mi.getProperties(sd).emission;
    }

    // Check whether to terminate based on max depth.
    if (pathData.length >= kMaxBounces + 1)
    {
        pathData.terminated = true;
        return;
    }

    // Create differentiable material parameters.
    BSDFParameters bsdfParams = getMaterialParams(sd.materialID, pathData.pixel);

    // Compute ray origin for new rays spawned from the hit.
    float3 rayOrigin = no_diff sd.computeRayOrigin();

    // Generate scatter ray for the next path segment.
    if (!generateScatterRay(sd, bsdfParams, rayOrigin, sg, pathData, ray))
    {
        pathData.terminated = true;
        return;
    }

    pathData.length++;
}

[Differentiable]
[PreferRecompute]
float3 tracePath(const uint2 pixel, const uint2 frameDim)
{
    PathData pathData = PathData(pixel);

    // Create sample generator.
    SampleGenerator sg = SampleGenerator(pixel, gFrameCount);

    const float3 primaryRayOrigin = gScene.camera.getPosition();
    const float3 primaryRayDir = gScene.camera.computeRayPinhole(pixel, frameDim).dir;
    Ray ray = Ray(primaryRayOrigin, primaryRayDir, 0.f, kRayTMax);
    HitInfo hit;

    // Handle primary hit.
    if (traceRayInline(ray, hit))
    {
        pathData.isHit = true;
    }
    else
    {
        pathData.terminated = true;
        pathData.isHit = false;
        pathData.radiance = kUseEnvBackground ? no_diff gScene.envMap.eval(primaryRayDir) : kDefaultBackgroundColor;
    }

    [MaxIters(MAX_BOUNCES + 2)]
    while (!pathData.terminated)
    {
        if (pathData.isHit)
        {
            handleHit(hit, sg, pathData, ray);

            if (pathData.terminated) break;

            pathData.isHit = traceRayInline(ray, hit);
        }
        else
        {
            // Add contribution from distant light (env map) in this direction.
            if (kUseEnvLight && (kComputeDirect || pathData.length > 0))
            {
                float3 Le = no_diff gScene.envMap.eval(ray.dir);
                pathData.radiance += pathData.thp * Le;
            }
            pathData.terminated = true;
        }
    }

    return pathData.radiance;
}

[shader("raygeneration")]
void rayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 frameDim = DispatchRaysDimensions().xy;

    // Primal rendering.
    float3 color = tracePath(pixel, frameDim);
    gOutputColor[pixel] = float4(color, 1.f);

    // Forward-mode differentiable rendering.
    DifferentialPair<float3> dpColor = fwd_diff(tracePath)(pixel, frameDim);
    gOutputFwdDColor[pixel] = float4(dpColor.d[0], 0.f, 0.f, 1.f);

    // Reverse-mode differentiable rendering.
    gOutputBwdDColor[pixel] = float4(0.f);
    bwd_diff(tracePath)(pixel, frameDim, float3(1.f, 0.f, 0.f));
}
