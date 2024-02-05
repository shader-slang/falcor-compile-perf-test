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
 * Compute pass that processes the trace results to advance the path or terminate.
 *
 * The host sets GROUP_SIZE_X and GROUP_SIZE_Y to set the thread group size.
 *
 * The macro FINAL_ITERATION is defined for the very last iteration when the
 * path has reached its max length. At that point, the path data is read-only
 * and no new rays will be generated.
 */
#include "Utils/Math/MathConstants.slangh"

import Scene.Shading;
import Scene.Material.ShadingUtils;
import Utils.Debug.PixelDebug;
import Internal.Utils.Sampling.SplitSampleGenerator;
import Rendering.Materials.InteriorListHelpers;
import Rendering.Utils.PixelStats;
import RenderPasses.LegacyPathTracer.PathTracer.PathTracerHelpers;
import RenderPasses.LegacyPathTracer.PathTracer.RayFootprint;
import RenderPasses.LegacyPathTracer.WavefrontPathTracer.PathStateHelpers;
import RenderPasses.PythonPathTracer.TraceReplayMaterials;
import RenderPasses.PythonPathTracer.HandleDisplacement;

struct Data
{
#ifdef FINAL_ITERATION
    // Use read-only resource on the final iteration to exploit faster data path (TLD vs SULD).
    // TODO: If we switch to root descriptors, there is no difference in the type of loads generated.
    StructuredBuffer<uint> pathFlags;
    StructuredBuffer<PathState> pathState;           ///< The path state.
    StructuredBuffer<PathRays> pathRays;             ///< The rays parameters generated at each path vertex.
    StructuredBuffer<RayFootprint> pathRayFootprint; ///< Ray footprint used for tex LOD.
    StructuredBuffer<InteriorList> pathInteriorList; ///< Path interior lists.
#else
    RWStructuredBuffer<uint> pathFlags;
    RWStructuredBuffer<PathState> pathState;           ///< The path state.
    RWStructuredBuffer<PathRays> pathRays;             ///< The rays parameters generated at each path vertex.
    RWStructuredBuffer<InteriorList> pathInteriorList; ///< Path interior lists.
    RWStructuredBuffer<RayFootprint> pathRayFootprint; ///< Ray footprint used for tex LOD.
    RWStructuredBuffer<float3> pathTransmission;       ///< The transmission ray parameters.
#endif
    RWStructuredBuffer<float3> pathOutput;   ///< Accumulated path contribution.
    StructuredBuffer<PackedHitInfo> pathHit; ///< Path hit info for the scatter ray (read only).

    // Samplers
    EnvMapSampler envMapSampler; ///< Environment map sampler.
};

// Static resources that don't change.
// TODO: Move into parameter block?
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

/**
 * Handles the case when a path missed the scene.
 * Depending on the configuration, we add the env map contribution here using MIS.
 * @param[in,out] pathData Current path data.
 */
void handleMiss(inout PathData path)
{
    // Check if the scatter event is samplable by the light sampling technique.
    const bool isLightSamplable = path.isLightSamplable();

    // If we have an environment, add it's weighted contribution here.
    if (gFurnaceTest)
    {
        path.L += path.thp;
    }
    else if (kUseEnvLight && (kUseLightsInDielectricVolumes || !path.isInsideVolume()) && (!kUseNEE || kUseMIS || !isLightSamplable))
    {
        logPathVertex();

        float misWeight = 1.f;
        if (kUseNEE && kUseMIS && isLightSamplable)
        {
            // We came here through BRDF sampling. The other sampling strategy is
            // env map sampling. Evaluate its probability for the current ray dir.
            float lightPdf = gData.envMapSampler.evalPdf(path.dir) * getEnvLightSelectionPdf();

            // Compute MIS weighted contribution from the environment map.
            misWeight = evalMIS(gParams, 1, path.pdf, kLightSamplesPerVertex, lightPdf);
        }

        float3 Le = gData.envMapSampler.eval(path.dir);

        // Accumulate emitted radiance weighted by path throughput.
        path.L += path.thp * Le * misWeight;
    }
}

/**
 * Handles the case when a path hit the scene.
 * Depending on the configuration, we add the emissive contribution here using MIS.
 * Then new shadow and scatter rays are generated.
 * @param[in] idx Path state index.
 * @param[in,out] pathData Current path data.
 */
void handleHit(const uint idx, inout PathData path)
{
    logPathVertex();

    // We only support triangle hits.
    const TriangleHit triangleHit = path.hit.getTriangleHit();
    const uint materialID = gScene.getMaterialID(triangleHit.instanceID);
    const bool isPython = gIsPythonMaterial[materialID];

    StaticVertexData triangleVertices[3];
    VertexData v = gScene.getVertexData(triangleHit, triangleVertices);

    // Update the ray footprint with current hit surface.
    path.rayFootprint.hitSurface(v, triangleVertices, triangleHit, path.origin, path.dir);

    // Create texture sampler based on current ray footprint for texLOD.
    let lod = path.rayFootprint.createTextureSampler(v, triangleVertices, triangleHit, path.origin, path.dir);

    // Prepare shading data.
    bool useNormalMap = !isPython || gUseNormalMap; // TODO: Cleanup
    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -path.dir, lod);
    // DisplacementContext displacement = makeDisplacementContext(sd, triangleHit);
    // if (!isPython) displacement.primaryHit(sd, -sd.V);

#ifndef FINAL_ITERATION
    if (kUseNestedDielectrics)
    {
        // Compute index of refraction for medium on the outside.
        sd.IoR = computeOutsideIoR(path.interiorList, sd.materialID, sd.frontFacing);

        // Handle volume absorption.
        if (!path.interiorList.isEmpty())
        {
            const uint interiorMaterialID = path.interiorList.getTopMaterialID();

            // Fetch volume absorption from the material. This field only exist in basic materials for now.
            if (gScene.materials.isBasicMaterial(interiorMaterialID))
            {
                BasicMaterialData md = gScene.materials.getBasicMaterialData(interiorMaterialID);

                float3 sigmaA = md.volumeAbsorption;
                float distance = length(path.origin - v.posW);
                path.thp *= exp(-sigmaA * distance);
            }
        }
    }
#endif

    // Create material instance.
    let mi = gScene.materials.getMaterialInstance(sd, lod);
    let bsdfProperties = mi.getProperties(sd);

    // Disable specular lobes if caustics are disabled and path already contains non-specular vertex.
    if (kDisableCaustics && path.nonSpecularBounces > 0 && bsdfProperties.roughness <= gParams.specularRoughnessThreshold)
    {
        sd.mtl.setActiveLobes((uint)LobeType::Diffuse);
    }

    // Check if the scatter event is samplable by the light sampling technique.
    const bool isLightSamplable = path.isLightSamplable();

    // Determine if we need to compute the emissive based on the current configuration.
    // It's only needed if emissive is enabled, and its full contribution hasn't been sampled elsewhere.
    const bool computeEmissive =
        kUseEmissiveLights && (kUseLightsInDielectricVolumes || !path.isInsideVolume()) && (!kUseNEE || kUseMIS || !isLightSamplable);
    const float3 emission = bsdfProperties.emission;

    if (computeEmissive && any(emission > 0.f))
    {
        float misWeight = 1.f;
        if (kUseNEE && kUseMIS && isLightSamplable)
        {
            // If emissive light sampling and MIS is enabled, we've already sampled emissive lights using NEE.
            // We need to evaluate the MIS weight here to account for the remaining contribution.
            // Note that MIS is only applied for hits on emissive triangles (other emissive geometry is not supported).

            // Prepare hit point struct with data needed for emissive light PDF evaluation.
            TriangleLightHit hit;
            hit.triangleIndex = gScene.lightCollection.getTriangleIndex(triangleHit.instanceID, triangleHit.primitiveIndex);
            hit.posW = sd.posW;
            hit.normalW = sd.frontFacing ? sd.faceN : -sd.faceN;

            // Evaluate PDF at the hit, had it been generated with light sampling.
            // Note that there is a minor discrepancy compared to BRDF sampling at the previous path vertex:
            // the normal passed through the ray payload is lossily compressed to reduce the ray payload size.
            float lightPdf = gEmissiveSampler.evalPdf(path.origin, path.normal, true, hit) * getEmissiveLightSelectionPdf();

            // Compute MIS weight by combining this with BRDF sampling.
            // Note we can assume path.pdf > 0.f since we shouldn't have got here otherwise.
            misWeight = evalMIS(gParams, 1, path.pdf, kLightSamplesPerVertex, lightPdf);
        }

        // Accumulate emitted radiance weighted by path throughput and MIS weight.
        path.L += path.thp * emission * misWeight;
    }

#ifndef FINAL_ITERATION
    // Unless path ends here, sample light and generate next path segment.
    path.length++;

    // Terminate after last path vertex has been processed.
    if (path.length > kMaxBounces || path.nonSpecularBounces > kMaxNonSpecularBounces)
        return;

    // Compute the ray origin for the next path segment.
    // The origin is offset slightly from the hit to avoid self-intersections.
    path.origin = sd.computeRayOrigin();

    // Split the random sequence with the light samples and the scatter ray.
#if USE_SPLIT_SAMPLE_GENERATOR
    var ssg = split4D(path.sg);
#else
#define ssg path.sg
#endif

    // Determine if material instance supports sampling with next-event estimation.
    bool supportsNEE = (mi.getLobeTypes(sd) & (uint)LobeType::NonDeltaReflection) != 0;

    // Generate shadow rays by sampling the scene lights.
    // We store the shadow rays directly to the path state buffer to reduce register pressure.
    if (kUseNEE && supportsNEE && (kUseLightsInDielectricVolumes || !path.isInsideVolume()))
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

    // Temporary keep path throughput from before the scatter event.
    const float3 prevThp = path.thp;

    // Save incoming ray direction for ray footprint.
    const float3 rayDirIn = path.dir;

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

    // Scatter the ray footprint out of the surface.
    path.rayFootprint.bounceOnSurface(
        v,
        triangleVertices,
        triangleHit,
        sd,
        v.normalW,
        rayDirIn,
        path.dir,
        bsdfProperties.roughness,
        path.isSpecular(),
        path.isTransmission()
    );

    // Check if this is the last path vertex.
    const bool isLastVertex = path.length == kMaxBounces || path.nonSpecularBounces > kMaxNonSpecularBounces;

    // Terminate if this is the last path vertex and light sampling already completely sampled incident radiance.
    if (isLastVertex && (kUseNEE && !kUseMIS && path.isLightSamplable()))
        path.clearScatterRay();

    // Terminate caustics paths.
    if (kDisableCaustics && path.isSpecular() && path.nonSpecularBounces > 0)
        path.clearScatterRay();
#endif
}

[numthreads(GROUP_SIZE_X, GROUP_SIZE_Y, 1)]
void processPath(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    printSetPixel(dispatchThreadId.xy);
    logSetPixel(dispatchThreadId.xy);

    const uint idx = getIdx(dispatchThreadId);
    if (idx == kInvalidIndex)
        return;

    // Load current path state.
    PathData path = loadPathState(idx, gData.pathFlags, gData.pathState, gData.pathRays, gData.pathInteriorList, gData.pathHit);

    // TODO: Use (kRayFootprintMode != TexLODMode::Mip0) when slang is fixed.
    if (!(kRayFootprintMode == TexLODMode::Mip0))
    {
        // Load ray footprint data.
        path.rayFootprint = gData.pathRayFootprint[idx];
    }

    if (path.isTerminated())
        return;

    // The contributions from the shadow rays have already been accumulated by loadPathState(). Just clear the flags.
    path.clearShadowRays();

    // Handle the hit/miss of the scatter ray. We'll generate new rays if it was a hit.
    if (path.hasScatterRay())
    {
        bool hit = path.flags & uint(PathFlags::scatterHit);
        path.clearScatterRay();

        if (hit && !gFurnaceTest)
        {
            handleHit(idx, path);
        }
        else
        {
            handleMiss(path);
        }

        // When we've reached the last path vertex, log the length of the path.
        if (!path.hasScatterRay())
            logPathLength(path.length);
    }

    // Accumulate contribution if it is nonzero.
    // TODO: Check that we don't stall here, if so we should move the load out of the conditional and higher up.
    if (any(path.L > 0.f))
    {
        gData.pathOutput[idx] += path.L;
    }

#ifndef FINAL_ITERATION
    // Store the updated path state.
    savePathState(path, idx, gData.pathFlags, gData.pathState, gData.pathRays, gData.pathInteriorList, gData.pathTransmission);

    // TODO: Use (kRayFootprintMode != TexLODMode::Mip0) when slang is fixed.
    if (!(kRayFootprintMode == TexLODMode::Mip0))
    {
        // Save ray footprint data.
        gData.pathRayFootprint[idx] = path.rayFootprint;
    }
#endif
}

void captureHit(const uint idx, in PathData path)
{
    // We only support triangle hits.
    const TriangleHit triangleHit = path.hit.getTriangleHit();
    const uint materialID = gScene.getMaterialID(triangleHit.instanceID);
    const bool isPython = gIsPythonMaterial[materialID];
    if (!isPython)
        return;

    StaticVertexData triangleVertices[3];
    VertexData v = gScene.getVertexData(triangleHit, triangleVertices);

    MaterialSampleGeometryInfo info;
    info.cameraPos = gScene.camera.getPosition();
    info.vertices = triangleVertices;
    info.v = v;
    info.lod = 0.0f;

    ExplicitLodTextureSampler lod = { 0.0f };

    bool useNormalMap = !isPython || gUseNormalMap; // TODO: Cleanup
    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -path.dir, lod);
    MaterialSample ms = MaterialSample(sd, info, sd.frame.toLocal(sd.V), false);

    let mi = gScene.materials.getMaterialInstance(sd, lod);

    // Compute the ray origin for the next path segment.
    // The origin is offset slightly from the hit to avoid self-intersections.
    path.origin = sd.computeRayOrigin();

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
    if (kUseNEE && (kUseLightsInDielectricVolumes || !path.isInsideVolume()))
    {
        for (uint i = 0; i < kLightSamplesPerVertex; i++)
        {
            let mat = TracingMaterialInstance(mi, ms, TracingMode::TraceEvalCalls, info, idx * kLightSamplesPerVertex + i);
            ShadowRay shadowRay = {};
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

[numthreads(GROUP_SIZE_X, GROUP_SIZE_Y, 1)]
void capturePath(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    printSetPixel(dispatchThreadId.xy);
    logSetPixel(dispatchThreadId.xy);

    const uint idx = getIdx(dispatchThreadId);
    if (idx == kInvalidIndex)
        return;

    // Load current path state.
    PathData path = loadPathState(idx, gData.pathFlags, gData.pathState, gData.pathRays, gData.pathInteriorList, gData.pathHit);
    if (path.isTerminated())
        return;

    // The contributions from the shadow rays have already been accumulated by loadPathState(). Just clear the flags.
    path.clearShadowRays();

    // Handle the hit/miss of the scatter ray. We'll generate new rays if it was a hit.
    if (path.hasScatterRay())
    {
        bool hit = path.flags & uint(PathFlags::scatterHit);
        path.clearScatterRay();

        if (hit)
        {
            captureHit(idx, path);
        }
    }
}
