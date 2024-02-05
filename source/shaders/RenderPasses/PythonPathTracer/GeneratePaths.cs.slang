/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/

/**
 * Compute pass that reads from the G/V-buffer and generates paths.
 *
 * The dispatch dimensions is one thread per pixel.
 * The program iterates over the samples to generate one path per sample.
 */
#include "Utils/Math/MathConstants.slangh"

import Utils.Debug.PixelDebug;
import Internal.Utils.Sampling.SplitSampleGenerator;
import Rendering.Materials.InteriorList;
import Rendering.Materials.InteriorListHelpers;
import Rendering.Utils.PixelStats;
import RenderPasses.LegacyPathTracer.PathTracer.PathTracerHelpers;
import RenderPasses.LegacyPathTracer.PathTracer.RayFootprint;
import RenderPasses.LegacyPathTracer.WavefrontPathTracer.PathStateHelpers;
import RenderPasses.PythonPathTracer.LoadShadingData;
import RenderPasses.PythonPathTracer.TraceReplayMaterials;
import RenderPasses.PythonPathTracer.HandleDisplacement;

struct Data
{
    // Outputs
    RWStructuredBuffer<uint> pathFlags;
    RWStructuredBuffer<PathState> pathState;           ///< The path state.
    RWStructuredBuffer<PathRays> pathRays;             ///< The rays parameters generated at each path vertex.
    RWStructuredBuffer<float3> pathTransmission;       ///< The transmission ray parameters.
    RWStructuredBuffer<InteriorList> pathInteriorList; ///< Path interior lists.
    RWStructuredBuffer<float3> pathOutput;             ///< Path contribution.
    RWStructuredBuffer<RayFootprint> pathRayFootprint; ///< Ray footprint used for tex LOD.
    StructuredBuffer<PackedHitInfo> pathHit;           ///< Path hit info for the scatter ray (read only).

    // Samplers
    EnvMapSampler envMapSampler; ///< Environment map sampler.
};

// Static resources that don't change.
// TODO: Move into parameter block
cbuffer StaticCB
{
    Data gData;
};

// Resources that may change per frame.
cbuffer PerFrameCB
{
    PathTracerParams gParams;
    EmissiveLightSampler gEmissiveSampler; ///< Emissive light sampler. TODO: Move to StaticCB?
}

// Optional outputs.
RWTexture2D<float4> gOutputAlbedo;
#define is_valid(name) (is_valid_##name != 0)

[numthreads(GROUP_SIZE_X, GROUP_SIZE_Y, 1)]
void generatePath(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    const uint2 pixel = dispatchThreadId.xy;

    printSetPixel(pixel);
    logSetPixel(pixel);

    // Mark paths for out-of-bounds pixels as terminated and exit.
    if (any(pixel >= gParams.frameDim))
    {
        // TODO: Remove the explicit clear of the flags for out-of-bounds pixels when indexing logic has been rewritten.
        // Clearing the buffer on the host was slightly slower, so keeping the code here for now.
        for (uint sampleIdx = 0; sampleIdx < kSamplesPerPixel; sampleIdx++)
        {
            uint idx = getIdx(uint3(pixel, sampleIdx));
            if (idx == kInvalidIndex)
                continue; // TODO: Change the indexing so that this check is not required

            gData.pathFlags[idx] = 0;
        }
        return;
    }

    // Prepare path state shared between all samples in the pixel.
    // All paths originate at the sample location, so we start by fetching material data.
    HitInfo hit;
    ShadingData sd;
    let lod = createTextureSampler(pixel);

    uint frameSeed = gParams.useFixedSeed ? 0 : gParams.frameCount;
    SampleGenerator sg = SampleGenerator(pixel, frameSeed * kSamplesPerPixel);
    const bool validPixel = loadShadingData(pixel, gParams.frameDim, gScene.camera, lod, sd, hit, sg);
    const bool isPython = gIsPythonMaterial[sd.materialID];
    // DisplacementContext displacement = makeDisplacementContext(sd, hit.getTriangleHit());
    // if (validPixel && !isPython) displacement.primaryHit(sd, -sd.V);

    PathData path = {};
    float3 L = float3(0);
    float3 outAlbedo = float3(0);
    IMaterialInstance mi;
    BSDFProperties bsdfProperties = {};

    if (validPixel)
    {
        // Create material instance and query its properties.
        mi = gScene.materials.getMaterialInstance(sd, lod);
        bsdfProperties = mi.getProperties(sd);

        // Always output directly emitted light from the primary hit.
        // This is independent of whether emissive materials are treated as light sources or not.
        L = bsdfProperties.emission;

        // Compute the ray origin for the new rays.
        path.origin = sd.computeRayOrigin();

        outAlbedo = bsdfProperties.diffuseReflectionAlbedo + bsdfProperties.specularReflectionAlbedo; // TODO: Include transmission?
    }
    else
    {
        // Background pixel.
        L = evalBackground(-sd.V);

        outAlbedo = L;
    }

    // Write per-pixel output. This is currently only albedo.
    if (is_valid(gOutputAlbedo))
        gOutputAlbedo[pixel] = float4(outAlbedo, 1);

    // Generate per-sample path state.
    // This includes sampling shadow and scatter rays for each path.
    for (uint sampleIdx = 0; sampleIdx < kSamplesPerPixel; sampleIdx++)
    {
        uint idx = getIdx(uint3(pixel, sampleIdx));
        if (idx == kInvalidIndex)
            continue; // TODO: Change the indexing so that this check is not required

        path.flags = 0;
        path.nonSpecularBounces = 0;
        path.thp = float3(0);
        path.L = L;

        if (validPixel)
        {
            logPathVertex();

            // Pixel represents a primary hit. We start a path here.
            path.thp = float3(1);

            // Reset interior list.
            if (kUseNestedDielectrics)
                path.interiorList = {};

            // Create sample generator.
            if (sampleIdx == 0)
            {
                path.sg = sg;
            }
            else
            {
                path.sg = SampleGenerator(pixel, frameSeed * kSamplesPerPixel + sampleIdx);
            }

            // Advance the generator to the first available dimension.
            // TODO: This is potentially expensive. We may want to store/restore the state from memory if it becomes a problem.
            for (uint i = 0; i < gParams.prngDimension; i++)
                sampleNext1D(path.sg);

                // Split the random sequence with the light samples and the scatter ray.
#if USE_SPLIT_SAMPLE_GENERATOR
            var ssg = split4D(path.sg);
#else
#define ssg path.sg
#endif
            // Determine if BSDF supports sampling with next-event estimation.
            bool supportsNEE = (mi.getLobeTypes(sd) & (uint)LobeType::NonDeltaReflection) != 0;

            // Generate shadow rays by sampling the scene lights.
            // We store the shadow rays directly to the path state buffer to reduce register pressure.
            // We don't handle kUseLightsInDielectricVolumes here, because we assume the primary hit is not inside a volume.
            if (kUseNEE && !gFurnaceTest && supportsNEE)
            {
                for (uint i = 0; i < kLightSamplesPerVertex; i++)
                {
                    ShadowRay shadowRay = {};
                    bool valid;
                    if (isPython)
                    {
                        let mat = ReplayMaterialInstance(mi, gPythonInference.mapIdx(idx * kLightSamplesPerVertex + i));
                        valid = generateShadowRay(gParams, gData.envMapSampler, gEmissiveSampler, sd, mat, i, path, ssg, shadowRay);
                    }
                    else
                    {
                        valid = generateShadowRay(gParams, gData.envMapSampler, gEmissiveSampler, sd, mi, i, path, ssg, shadowRay);
                        // if (displacement.secondaryHit(shadowRay.rayParams.xyz)) shadowRay.Lr = float3(0.0f);
                    }
                    if (valid)
                    {
                        gData.pathRays[idx].shadowRay[i] = shadowRay.rayParams;
                        gData.pathState[idx].Lr[i] = shadowRay.Lr;
                    }
#if USE_SPLIT_SAMPLE_GENERATOR
                    ssg.nextInstance();
#endif
                }
            }

            // Generate the next path segment.
            if (isPython)
            {
                let mat = ReplayMaterialInstance(mi, gPythonSample.mapIdx(idx));
                generateScatterRay(gParams, sd, mat, path, ssg);
            }
            else
            {
                generateScatterRay(gParams, sd, mi, path, ssg);
                // if (displacement.secondaryHit(path.dir)) path.thp = float3(0.0f);
            }

            // Check if this is the last path vertex.
            const bool isLastVertex = path.length == kMaxBounces || path.nonSpecularBounces > kMaxNonSpecularBounces;

            // Generate ray footprint data for Tex LOD.
            // TODO: Use (kRayFootprintMode != TexLODMode::Mip0) when slang is fixed.
            if (!(kRayFootprintMode == TexLODMode::Mip0))
            {
                // Ray footprint initalization (TexLOD). Uses vbuffer parameters fetched by loadShadingData.
                gData.pathRayFootprint[idx] = RayFootprint::create(
                    hit.getTriangleHit(),
                    pixel,
                    gParams.frameDim,
                    sd.posW.xyz,
                    gScene.camera.getPosition(),
                    sd.faceN,
                    bsdfProperties.guideNormal,
                    gParams.screenSpacePixelSpreadAngle,
                    bsdfProperties.roughness,
                    path.isSpecular()
                );
            }

            // Terminate if this is the last path vertex and light sampling already completely sampled incident radiance.
            if (isLastVertex && (kUseNEE && !gFurnaceTest && !kUseMIS && path.isLightSamplable()))
                path.clearScatterRay();
        }

        // Store the new path state and contribution.
        savePathState(path, idx, gData.pathFlags, gData.pathState, gData.pathRays, gData.pathInteriorList, gData.pathTransmission);

        gData.pathOutput[idx] = path.L;
    }
}

[numthreads(GROUP_SIZE_X, GROUP_SIZE_Y, 1)]
void capturePath(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    const uint2 pixel = dispatchThreadId.xy;
    printSetPixel(pixel);
    logSetPixel(pixel);

    if (any(pixel >= gParams.frameDim))
        return;

    // Prepare path state shared between all samples in the pixel.
    // All paths originate at the sample location, so we start by fetching material data.
    HitInfo hit;
    ShadingData sd;
    let lod = createTextureSampler(pixel);

    uint frameSeed = gParams.useFixedSeed ? 0 : gParams.frameCount;
    SampleGenerator sg = SampleGenerator(pixel, frameSeed * kSamplesPerPixel);
    const bool validPixel = loadShadingData(pixel, gParams.frameDim, gScene.camera, lod, sd, hit, sg);
    if (!validPixel)
        return;
    const bool isPython = gIsPythonMaterial[sd.materialID];
    if (!isPython)
        return;

    StaticVertexData vertices[3];
    VertexData v = gScene.getVertexData(hit.getTriangleHit(), vertices);

    MaterialSampleGeometryInfo info;
    info.cameraPos = gScene.camera.getPosition();
    info.vertices = vertices;
    info.v = v;
    info.lod = 0.0f;

    MaterialSample ms = MaterialSample(sd, info, sd.frame.toLocal(sd.V), false);

    IMaterialInstance mi = gScene.materials.getMaterialInstance(sd, lod);

    // Compute the ray origin for the new rays.
    PathData path = {};
    path.origin = sd.computeRayOrigin();

    // Generate per-sample path state.
    // This includes sampling shadow and scatter rays for each path.
    for (uint sampleIdx = 0; sampleIdx < kSamplesPerPixel; sampleIdx++)
    {
        uint idx = getIdx(uint3(pixel, sampleIdx));
        if (idx == kInvalidIndex)
            continue;

        path.flags = 0;
        path.nonSpecularBounces = 0;
        path.thp = float3(1);

        // Create sample generator.
        if (sampleIdx == 0)
        {
            path.sg = sg;
        }
        else
        {
            path.sg = SampleGenerator(pixel, frameSeed * kSamplesPerPixel + sampleIdx);
        }

        // Advance the generator to the first available dimension.
        // TODO: This is potentially expensive. We may want to store/restore the state from memory if it becomes a problem.
        for (uint i = 0; i < gParams.prngDimension; i++)
            sampleNext1D(path.sg);

            // Split the random sequence with the light samples and the scatter ray.
#if USE_SPLIT_SAMPLE_GENERATOR
        var ssg = split4D(path.sg);
#else
#ifndef ssg
#define ssg path.sg
#endif
#endif
        // Generate shadow rays by sampling the scene lights.
        // We store the shadow rays directly to the path state buffer to reduce register pressure.
        // We don't handle kUseLightsInDielectricVolumes here, because we assume the primary hit is not inside a volume.
        if (kUseNEE && !gFurnaceTest)
        {
            for (uint i = 0; i < kLightSamplesPerVertex; i++)
            {
                ShadowRay shadowRay = {};
                let mat = TracingMaterialInstance(mi, ms, TracingMode::TraceEvalCalls, info, idx * kLightSamplesPerVertex + i);
                generateShadowRay(gParams, gData.envMapSampler, gEmissiveSampler, sd, mat, i, path, ssg, shadowRay);
#if USE_SPLIT_SAMPLE_GENERATOR
                ssg.nextInstance();
#endif
            }
        }

        // Generate the next path segment.
        let mat = TracingMaterialInstance(mi, ms, TracingMode::TraceSampleOrPdfCalls, info, idx);
        generateScatterRay(gParams, sd, mat, path, ssg);
    }
}
