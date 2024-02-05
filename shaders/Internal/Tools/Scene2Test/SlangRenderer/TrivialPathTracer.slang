/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
import Internal.Scene2.Scene2;
import Internal.Scene2.HitInfo;
import Internal.Scene2.VertexData;

import Utils.Math.Ray;
import Utils.Math.MathHelpers;

__exported import Scene.ShadingData;
__exported import Scene.Material.TextureSampler;
__exported import Scene.Material.MaterialFactory;

__exported import Utils.Sampling.SampleGenerator;

#define M_PI 3.14159265358979323846    // pi
#define M_1_PI 0.318309886183790671538 // 1/pi
#define M_2PI 6.28318530717958647693   // 2pi

float3 sampleCosineHemisphere(float2 u, out float pdf)
{
    float3 p;
    float r = sqrt(u.x);
    float phi = M_2PI * u.y;
    p.x = r * cos(phi);
    p.y = r * sin(phi);
    p.z = sqrt(1.f - u.x);
    pdf = p.z * M_1_PI;
    return p;
}

/**
 * @brief The basic prototype PathTracer that handles path hits and misses, updating its path state.
 *
 * The misses are assumed to hit env map with a black-to-white gradient,
 * while hits assume that the material is Lambert with color given by the VertexData.
 */
struct TrivialPathTracer
{
    struct PathState
    {
        Ray ray; ///< Hit ray on entry into hit handling, scatter ray afterwards
        uint isActive;
        float3 throughput;
        float3 totalContribution;
        SampleGenerator sg;

        __init(Ray _ray, uint seed)
        {
            isActive = 1;
            throughput = float3(1.f);
            totalContribution = float3(0.f);
            sg = SampleGenerator(seed);
            ray = _ray;
        }
    }

    void handleInvalid(inout PathState path) { path.isActive = 0; }

    void handleMiss(inout PathState path)
    {
        float3 envMapColor;
        if (gScene.hasEnvMap)
        {
            envMapColor = gScene.envMap.eval(path.ray.dir);
        }
        else
        {
            float gradient = (path.ray.dir.y + 1.f) * 0.5f;
            const float3 bottom = float3(0.0f);
            const float3 top = float3(1.0f);
            envMapColor = lerp(bottom, top, float3(saturate(gradient)));
        }
        path.totalContribution += path.throughput * envMapColor;
        path.isActive = 0;
    }

    void handleHit<THitInfo : IHitInfo>(inout PathState path, const THitInfo hitInfo)
    {
        VertexData v;
        if (!hitInfo.getVertexData(path.ray, v))
            return handleMiss(path);

        float3 N = dot(path.ray.dir, v.normalW) > 0.f ? -v.normalW : v.normalW;
        float3 T, B;
        branchlessONB(N, B, T);

        uint materialID = gScene.geometryData[hitInfo.getGeometryDataIndex()].materialID;

        float pdf;
        float3 dirW;
        if (materialID != uint(-1) && materialID < gScene.materials.getMaterialCount())
        {
            let lod = ExplicitLodTextureSampler(0.f);
            const ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -path.ray.dir, lod);
            // This really belongs more into the material instance itself
            uint hints = 0;
            if (any(v.normalW != v.faceNormalW))
                hints |= (uint)MaterialInstanceHints::AdjustShadingNormal;

            const IMaterialInstance mi = gScene.materials.getMaterialInstance(sd, lod, hints);
            BSDFSample bsdfSample;
            bool valid = mi.sample(sd, path.sg, bsdfSample);
            if (!valid)
                return handleInvalid(path);
            dirW = bsdfSample.wo;
            path.throughput *= bsdfSample.weight;
        }
        else
        {
            float3 color = { 0.18f, 0.18f, 0.18f };
            if (materialID != uint(-1))
                color = { 0.8f, 0.05f, 0.8f };
            float3 dir = sampleCosineHemisphere(sampleNext2D(path.sg), pdf);
            dirW = dir.x * T + dir.y * B + dir.z * N;
            path.throughput *= color;
        }

        path.ray = hitInfo.makeScatterRay(path.ray, v, dirW);
    }
}

TrivialPathTracer gPathTracer;
