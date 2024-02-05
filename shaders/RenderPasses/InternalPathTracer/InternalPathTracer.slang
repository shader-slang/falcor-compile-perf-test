/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
import Rendering.Lights.EnvMapSampler;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Rendering.Lights.LightHelpers;
import Rendering.Materials.Fresnel;
import Rendering.Materials.IsotropicGGX;
import Rendering.Materials.TexLODHelpers;
import Rendering.Materials.InteriorListHelpers;
import Rendering.Volumes.GridVolumeSampler;
import Rendering.Volumes.HomogeneousVolumeSampler;
import Rendering.Volumes.PhaseFunction;
import Rendering.Utils.PixelStats;
import Utils.Color.ColorHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Debug.PixelDebug;
import Utils.Debug.WarpProfiler;
import RenderPasses.Shared.Denoising.NRDConstants;
import RenderPasses.Shared.Denoising.NRDBuffers;
import RenderPasses.Shared.Denoising.NRDHelpers;
import ScreenSpaceReSTIR.ScreenSpaceReSTIR;
import NeuralRadianceCache.NeuralRadianceCache;
import LoadShadingData;
import ColorType;
import NRDHelpers;

__exported import RenderPasses.InternalPathTracer.PathState; // TODO: Using full search path as workaround for Slang issue (#105), remove when fixed.
__exported import PathCoherenceHints;
__exported import Params;

/** Interface for querying the closest hit in the scene.
    This is used in `nextHit` to allow using different raytracing APIs.
*/
interface IClosestHitQuery
{
    /** Trace a ray against the scene and return the closest hit point.
        Note: The content of the path state must not be changed in the implementation of this interface.
        However, for the ReorderTraceRay scheduler it is necessary to pass the state through to
        TraceRayHitObject as ray payload and then re-materialize it after ray traversal.
        \param[in,out] path Path state.
        \param[in] ray Ray.
        \param[out] hit Hit info (only valid if function returns true).
        \param[out] hitT Hit distance (only valid if function returns true).
        \return Returns true if the ray does intersect the scene.
    */
    [mutating] bool traceRay(inout PathState path, const Ray ray, out HitInfo hit, out float hitT);
};

/** Interface for querying visibility in the scene.
    This is used in `handleHit` (and other variants) to allow using different raytracing APIs.
*/
interface IVisibilityQuery
{
    /** Trace a visibility ray against the scene.
        \param[in] ray Ray.
        \return Returns true if the ray endpoints are mutually visible (i.e. the ray does NOT intersect the scene).
    */
    [mutating] bool traceVisibilityRay(const Ray ray);
};

/** Internal path tracer.

    This implements the high-level path tracing logic and is shared among
    different scheduling implementations.
*/
struct InternalPathTracer
{
#ifndef ENABLE_STATS
#error ENABLE_STATS is not defined.
#endif
    /// Helper for collecting warp profiling statistics, if enabled.
    void warpProfile(WarpProfilerEvent event, uint pathVertexIndex, uint materialID)
    {
#if ENABLE_STATS
        uint vtxIdx = clamp(pathVertexIndex, 1, kMaxWarpProfilerVertices); // Collecting stats for vertices 1...N where N=kMaxWarpProfilerVertices.
        uint binIdx = (uint)event * 2 * kMaxWarpProfilerVertices + vtxIdx - 1;
        warpUtilization(binIdx);
        warpDivergence(binIdx + kMaxWarpProfilerVertices, materialID);
#endif
    }

    PathTracerParams params;                            ///< Runtime parameters.

    // Samplers
    EnvMapSampler envMapSampler;                        ///< Environment map sampler. Only valid when kUseEnvLight == true.
    EmissiveLightSampler emissiveSampler;               ///< Emissive light sampler. Only valid when kUseEmissiveLights == true.
    ScreenSpaceReSTIR screenSpaceReSTIR;                ///< Screen space ReSTIR sampler if kUseScreenSpaceReSTIR == true.
    NRC nrc;                                            ///< Neural radiance cache. Only valid if kUseNRC == true.
    GridVolumeSampler gridVolumeSampler;                ///< Grid volume sampler. Only valid when kUseGridVolumes == true.

    // Inputs
    Texture2D<PackedHitInfo> vbuffer;                   ///< Fullscreen V-buffer for the primary hits.
    Texture2D<float> spatioTemporalBlueNoise;           ///< Spatio-temporal blue noise table
    Texture2D<float3> viewDir;                          ///< Optional view direction. Only valid when kUseViewDir == true.
    Texture2D<uint> sampleCount;                        ///< Optional input sample count buffer. Only valid when kSamplesPerPixel == 0.
    Texture2D<uint> sampleOffset;                       ///< Output offset into per-sample buffers. Only valid when kSamplesPerPixel == 0.

    RWStructuredBuffer<ColorType> sampleColor;          ///< Output per-sample color if kSamplesPerPixel != 1.
    RWStructuredBuffer<GuideData> sampleGuideData;      ///< Output per-sample guide data.
    NRDBuffers outputNRD;                               ///< Output NRD data.

    RWTexture2D<float4> outputColor;                    ///< Output color buffer if kSamplesPerPixel == 1.
    RWTexture2D<float4> outputDebug;                    ///< Output debug data. Valid if OUTPUT_DEBUG is nonzero.
    RWTexture2D<uint> outputTime;                       ///< Output timing data. Valid if OUTPUT_TIME is nonzero.

    /*******************************************************************
                                Static members
    *******************************************************************/

    // Render settings that depend on the scene.
    // TODO: Move into scene defines.
    static const bool kUseEnvLight = USE_ENV_LIGHT;
    static const bool kUseEmissiveLights = USE_EMISSIVE_LIGHTS;
    static const bool kUseAnalyticLights = USE_ANALYTIC_LIGHTS;
    static const bool kUseGridVolumes = USE_GRID_VOLUMES;
    static const bool kUseCurves = USE_CURVES;
    static const bool kUseHairMaterial = USE_HAIR_MATERIAL;
#if defined(HAS_MATERIAL_VOLUME_PROPERITES)
    static const bool kSupportPathStateVolumeProperties = true;
#else
    static const bool kSupportPathStateVolumeProperties = false;
#endif

    // Additional specialization.
    static const bool kOutputGuideData = OUTPUT_GUIDE_DATA;
    static const bool kOutputTime = OUTPUT_TIME;
    static const bool kEnableStats = ENABLE_STATS;

    // Stochastic texture filtering specialization
    static const bool kUseStochasticBicubicFiltering = USE_STOCHASTIC_BICUBIC_FILTERING;
    static const bool kUseSpatioTemporalBlueNoise = USE_SPATIO_TEMP_BLUE_NOISE;

    /** Types of samplable lights.
    */
    enum class LightType
    {
        EnvMap,
        Emissive,
        Analytic
    };

    /** Describes a light sample.
    */
    struct LightSample
    {
        float3  Li;         ///< Incident radiance at the shading point (unshadowed). This is already divided by the pdf.
        float   pdf;        ///< Pdf with respect to solid angle at the shading point.
        float3  origin;     ///< Ray origin for visibility evaluation (offseted to avoid self-intersection).
        float   distance;   ///< Ray distance for visibility evaluation (shortened to avoid self-intersection).
        float3  dir;        ///< Ray direction for visibility evaluation (normalized).
        uint    lightType;  ///< Light type this sample comes from (LightType casted to uint).

        Ray getVisibilityRay() { return Ray(origin, dir, 0.f, distance); }
    };

    /** Describes a path vertex.
    */
    struct PathVertex
    {
        uint index;         ///< Vertex index (0 = camera, 1 = primary hit, 2 = secondary hit, etc.).
        float3 pos;         ///< Vertex position.
        float3 faceNormal;  ///< Geometry normal at the vertex (zero if not on a surface).
        bool frontFacing;   ///< True if path vertex is on the front-facing side (if on a surface).

        /** Initializes a path vertex.
            \param[in] index Vertex index.
            \param[in] pos Vertex position.
            \param[in] faceNormal Geometry normal.
            \param[in] frontFacing Front-facing flag.
        */
        __init(uint index, float3 pos, float3 faceNormal = float3(0.f), bool frontFacing = true)
        {
            this.index = index;
            this.pos = pos;
            this.faceNormal = faceNormal;
            this.frontFacing = frontFacing;
        }

        /** Get position with offset applied in direction of the geometry normal to avoid self-intersection
            for visibility rays.
            \param[in] rayDir Direction of the visibility ray (does not need to be normalized).
            \return Returns the offseted position.
        */
        float3 getRayOrigin(float3 rayDir)
        {
            return computeRayOrigin(pos, dot(faceNormal, rayDir) >= 0 ? faceNormal : -faceNormal);
        }

        /** Returns the oriented face normal.
            \return Face normal flipped to the same side as the view vector.
        */
        float3 getOrientedFaceNormal()
        {
            return frontFacing ? faceNormal : -faceNormal;
        }
    };

    /** Classify a hit to be handled on the specular queue.
    */
    static bool useSpecularQueue(const HitInfo hit)
    {
        // Only use specular queue for triangle hits.
        if (hit.getType() != HitType::Triangle) return false;

        const TriangleHit triangleHit = hit.getTriangleHit();

        // Check if the material is compatible with the specular queue.
        // Currently that requires purely delta specular reflection/transmission.
        // We inspect the material directly to avoid having to fetch and interpolate the vertices.
        uint materialID = gScene.getMaterialID(triangleHit.instanceID);
        MaterialHeader materialHeader = gScene.materials.getMaterialHeader(materialID);
        return materialHeader.isDeltaSpecular();
    }

    /** Set guiding data when background is hit.
    */
    static void setBackgroundGuideData(inout GuideData guideData, const float3 dir, const float3 Le)
    {
        if (kOutputGuideData)
        {
            guideData.setGuideNormal(-dir);
            // Compress dynamic range similar to UE4.
            const float3 compressedColor = pow(Le / (Le + 1.f), 0.454545f);
            guideData.setAlbedo(compressedColor);
            guideData.setSpecularAlbedo(float3(0.f));
            guideData.setIndirectAlbedo(float3(0.f));
            guideData.setReflectionPos(float3(0.f));
            guideData.specHitDist = 0.f;
        }
    }

    /** Set guiding data when primary surface is hit.
    */
    static void setPrimarySurfaceGuideData(inout GuideData guideData, const ShadingData sd, const BSDFProperties bsdfProperties)
    {
        if (kOutputGuideData)
        {
            guideData.setGuideNormal(bsdfProperties.guideNormal);
            // Use sum of reflection/transmission albedo as they are denoised together.
            guideData.setAlbedo(bsdfProperties.diffuseReflectionAlbedo + bsdfProperties.diffuseTransmissionAlbedo);
            guideData.setReflectionPos(sd.posW);

            const float NdotV = saturate(dot(bsdfProperties.guideNormal, sd.V));
            const float ggxAlpha = bsdfProperties.roughness * bsdfProperties.roughness;

            const bool isHairMaterial = kUseHairMaterial && (sd.mtl.getMaterialType() == MaterialType::Hair);

            // We use an approximate shading model with a single specular lobe.
            // TODO: Generalize this to arbitrary materials.
            if (isHairMaterial)
            {
                // Use a simple heuristic for hair material.
                const float3 preintegratedSpecular = approxSpecularIntegralGGX(float3(0.04f), ggxAlpha, NdotV);
                guideData.setSpecularAlbedo(preintegratedSpecular);
            }
            else
            {
                // Use pre-integrated FG term for specular
                const float3 preintegratedSpecular = approxSpecularIntegralGGX(bsdfProperties.specularReflectance, ggxAlpha, NdotV);
                guideData.setSpecularAlbedo(preintegratedSpecular);
            }
        }
    }

    /** Set guiding data when indirect surface is hit.
    */
    static void setIndirectSurfaceGuideData(inout GuideData guideData, const ShadingData sd, const BSDFProperties bsdfProperties)
    {
        if (any(bsdfProperties.emission > 0.f))
        {
            float3 indirectBaseColor = pow(bsdfProperties.emission / (bsdfProperties.emission + 1.0f), 0.454545f);
            guideData.setIndirectAlbedo(indirectBaseColor);
        }
        else
        {
            // Use sum of reflection/transmission albedo as they are denoised together.
            guideData.setIndirectAlbedo(bsdfProperties.diffuseReflectionAlbedo + bsdfProperties.diffuseTransmissionAlbedo);
        }
        guideData.setReflectionPos(sd.posW);
    }

    /*******************************************************************
                              Member functions
    *******************************************************************/

    /** Check if the path has finished all surface bounces and needs to be terminated.
        Note: This is expected to be called after generateScatterRay(), which increments the bounce counters.
        \param[in] path Path state.
        \return Returns true if path has processed all bounces.
    */
    bool hasFinishedSurfaceBounces(const PathState path)
    {
        const uint diffuseBounces = path.getBounces(BounceType::Diffuse);
        const uint specularBounces = path.getBounces(BounceType::Specular);
        const uint transmissionBounces = path.getBounces(BounceType::Transmission);
        const uint surfaceBounces = diffuseBounces + specularBounces + transmissionBounces;
        return
            (surfaceBounces > kMaxSurfaceBounces) ||
            (diffuseBounces > kMaxDiffuseBounces) ||
            (specularBounces > kMaxSpecularBounces) ||
            (transmissionBounces > kMaxTransmissionBounces);
    }

    /** Compute the total length of a terminated path.
        \param[in] path Path state.
        \return Returns the total number of bounces a path took.
    */
    uint getTerminatedPathLength(const PathState path)
    {
        // Account for the fact that we may have counted one bounce too many (scatter ray at the last path vertex).
        uint diffuseBounces = min(kMaxDiffuseBounces, path.getBounces(BounceType::Diffuse));
        uint specularBounces = min(kMaxSpecularBounces, path.getBounces(BounceType::Specular));
        uint transmissionBounces = min(kMaxTransmissionBounces, path.getBounces(BounceType::Transmission));
        uint surfaceBounces = min(kMaxSurfaceBounces, diffuseBounces + specularBounces + transmissionBounces);
        uint volumeBounces = min(kMaxVolumeBounces, path.getBounces(BounceType::Volume));
        return surfaceBounces + volumeBounces;
    }

    /** Generate the path state for specular path.
        \param[in] packedPath Packed specular path.
        \param[out] path Path state for specular path.
    */
    void generateSpecularPath(const PackedSpecularPath packedPath, inout PathState path)
    {
        path.decodeSpecularPath(packedPath);
        if (kUseScreenSpaceReSTIR && ScreenSpaceReSTIR::kUseGI)
        {
            path.reSTIRGIData = {};
            path.reSTIRGIData.L0 = 0;
            path.reSTIRGIData.isDisabled = (kSamplesPerPixel > 1 || path.getVertexIndex() > 1);
        }
    }

    struct GeneratePathClosestHitQuery : IClosestHitQuery
    {
        HitInfo primaryHit;
        float primaryHitT;

        bool traceRay(inout PathState path, const Ray ray, out HitInfo hit, out float hitT)
        {
            hit = {};
            hitT = {};

            if (primaryHit.isValid() && primaryHitT < ray.tMax)
            {
                hit = primaryHit;
                hitT = primaryHitT;
                return true;
            }
            return false;
        }
    };

    /** Generate the path state for a primary hit in screen space.
        This is treated equivalent to subsequent path vertices to reduce code divergence.
        \param[in] pathID Path ID which encodes pixel and sample index.
        \param[out] path Path state for the primary hit.
    */
    void generatePath(const uint pathID, out PathState path)
    {
        path = {};
        path.setActive();
        path.id = pathID;
        path.thp = float3(1.f);
        path.imageXform = float4x4(
            1.f, 0.f, 0.f, 0.f,
            0.f, 1.f, 0.f, 0.f,
            0.f, 0.f, 1.f, 0.f,
            0.f, 0.f, 0.f, 1.f);

        const uint2 pixel = path.getPixel();

        // When using the neural radiance cache, only training paths have to be traced fully.
        // The other paths can be cut short once we reach the prediction vertex index.
        if (kUseNRC)
        {
            path.nrcData.init(nrc.kMaxPathBounces);

            const uint3 pathOffset = uint3(pixel.x, pixel.y, path.getSampleIdx()) % nrc.kTrainingStride;
            path.nrcData.setForTraining(all(pathOffset == nrc.kTrainingOffset));

            // Get a pseudorandom selection of "unbiased" training paths. Unbiased means that the paths are traced to their full length.
            // tdavidovic: I would really like to see some () here, especially with the int divs and modulo (also, int divs, meh)
            path.nrcData.setUnbiased(path.nrcData.isForTraining() && pixel.x / nrc.kTrainingStride.x % 4 == nrc.kTrainingOffset.x % 4 && pixel.y / nrc.kTrainingStride.y % 4 == nrc.kTrainingOffset.y % 4);
            path.nrcData.setExitedScene(false);
        }

        // Create primary ray.
        Ray cameraRay = gScene.camera.computeRayPinhole(pixel, params.frameDim);
        if (kUseViewDir) cameraRay.dir = -viewDir[pixel];
        path.origin = cameraRay.origin;
        path.dir = cameraRay.dir;

        // Create sample generator.
        const uint maxSpp = kSamplesPerPixel > 0 ? kSamplesPerPixel : kMaxSamplesPerPixel;
        path.sg = SampleGenerator(pixel, params.seed * maxSpp + path.getSampleIdx());

        // Load the primary hit info from the V-buffer.
        const HitInfo hit = HitInfo(vbuffer[pixel]);

        // Grid volumes are currently not part of the scene geometry.
        // In order to check for volume scattering events *before* the primary hit,
        // we perform a vertex query and use the primary hit as the result of a "simulated" ray query.
        if (kUseGridVolumes)
        {
            GeneratePathClosestHitQuery chq = {};
            if (hit.isValid())
            {
                let lod = ExplicitLodTextureSampler(0.f);
                const ShadingData sd = loadShadingData(hit, path.origin, path.dir, lod);

                // Create material instance and query its properties.
                let hints = getMaterialInstanceHints(hit, true /* primary hit */);
                let mi = gScene.materials.getMaterialInstance(sd, lod, hints);
                let bsdfProperties = mi.getProperties(sd);
                warpProfile(WarpProfilerEvent::MaterialCreate, path.getVertexIndex(), sd.materialID);

                // Write primary surface guide data. If grid volumes are disabled, this is done in `handleSurfaceHit`.
                setPrimarySurfaceGuideData(path.guideData, sd, bsdfProperties);

                chq.primaryHit = hit;
                chq.primaryHitT = length(sd.posW - path.origin);
            }
            nextHit(path, chq);
        }
        else
        {
            // If invalid, the path is still active and treated as a miss.
            if (hit.isValid())
            {
                path.setHit(hit);
                path.setVertexIndex(1);
            }
        }

        // Setup ReSTIR GI.
        if (kUseScreenSpaceReSTIR && ScreenSpaceReSTIR::kUseGI)
        {
            path.reSTIRGIData = {};
            path.reSTIRGIData.L0 = 0;
            path.reSTIRGIData.isDisabled = (kSamplesPerPixel > 1 || path.getVertexIndex() > 1);
        }
    }

    /** Set up path for logging and debugging.
        \param[in] path Path state.
    */
    void setupPathLogging(const PathState path)
    {
        printSetPixel(path.getPixel());
        logSetPixel(path.getPixel());
    }

    /** Compute a set of coherence hints for a given path state.
        These hints predict the code path taken in the next call to handleHit() with the given path state.
        They are used by reordering schedulers to extract coherence.
        \param[in] path Path state.
        \param[in] isDeltaSpecular Hit material is pure delta specular.
        \param[in] isEmissive Hit material is emissive.
        \return Returns a set of hints.
    */
    PathCoherenceHints getCoherenceHints(const PathState path, const bool isDeltaSpecular, const bool isEmissive)
    {
        PathCoherenceHints hints = {};

        // Determines if handleHit() will return early due to russian roulette.
        if (kUseRussianRoulette)
        {
            // This is under the assumption that the first random number drawn from the sample generator is used for russian roulette.
            // Keep in sync with handleHit().
            PathState tmpPath = path;
            hints.terminatedByRussianRoulette = terminatePathByRussianRoulette(tmpPath, sampleNext1D(tmpPath.sg));
        }

        // Determines if handleHit() will return early.
        hints.terminated = hasFinishedSurfaceBounces(path);
        if (kDisableCaustics && path.getBounces(BounceType::Diffuse) > 0 && isDeltaSpecular) hints.terminated = true;

        // Determines if handleHit() will sample a light.
        hints.samplesLight = true;
        if (isDeltaSpecular) hints.samplesLight = false;
        if (!kUseLightsInDielectricVolumes && path.isInsideDielectricVolume()) hints.samplesLight = false;

        // Determines if handleHit() will compute emission.
        hints.computesEmissive = isEmissive && path.isLightSamplable();
        if (kDisableDirectIllumination && path.getVertexIndex() == 2) hints.computesEmissive = false;
        if (kUseScreenSpaceReSTIR && path.getVertexIndex() == 2 && !isDeltaSpecular) hints.computesEmissive = false;

        return hints;
    }

    /** Update the path throughouput.
        \param[in,out] path Path state.
        \param[in] weight Vertex throughput.
    */
    void updatePathThroughput(inout PathState path, const float3 weight)
    {
        if (kUseNRC && path.nrcData.isForTraining() && path.nrcData.vertexCount > 0)
        {
            path.nrcData.trainingVertexThroughput *= weight;
        }
        else
        {
            path.thp *= weight;
        }
    }

    /** Add radiance to the path contribution.
        \param[in,out] path Path state.
        \param[in] radiance Vertex radiance.
    */
    void addToPathContribution(inout PathState path, const float3 radiance)
    {
        if (kUseNRC && path.nrcData.isForTraining() && path.nrcData.vertexCount > 0)
        {
            path.nrcData.trainingVertexRadiance += path.nrcData.trainingVertexThroughput * radiance;
        }
        else
        {
            path.L += path.thp * radiance;
        }
    }

    /** DEMO21: Evaluate global light profile for materials that have it enabled.
    */
    void evaluateLightProfile(inout BSDFProperties bsdfProperties, const ShadingData sd, const PathState path)
    {
        if (Scene::kUseLightProfile && gScene.materials.getMaterialHeader(sd.materialID).isLightProfileEnabled())
        {
            // Do not apply light profile on primary hits in order to have full emission in all directions.
            if (path.getVertexIndex() > 0)
            {
                bsdfProperties.emission *= gScene.lightProfile.eval(dot(sd.faceN, -path.dir));
            }
        }
    }

    void setVolumePropertiesOnPath(const IMaterialInstance mi, const bool frontFacing, const uint lobeType, inout PathState path )
    {
        // Initialize volume parameters if necessary. This will only work for one volume.
        if ( kSupportPathStateVolumeProperties && mi.hasVolumeProperties() && frontFacing && (lobeType & (uint)LobeType::Transmission) != 0 )
        {
            VolumeProperties vp = mi.getVolumeProperties();
            path.sigmaA = vp.sigmaA;
            path.sigmaS = vp.sigmaS;
            path.setHasVolumeProperties(true);
        }
        else if ( (!mi.hasVolumeProperties() || !frontFacing) && (lobeType & (uint)LobeType::Transmission) != 0)
        {
            path.setHasVolumeProperties(false);
        }
        // else keep exsiting parameters (i.e we reflect on the inside)
    }


    bool generateScatterRay(const ShadingData sd, const IMaterialInstance mi, inout PathState path)
    {
        BSDFSample result;
        bool valid = mi.sample(sd, path.sg, result, kUseBSDFSampling);
        warpProfile(WarpProfilerEvent::MaterialSample, path.getVertexIndex(), sd.materialID);

        return generateScatterRay(result, sd, mi, path, valid);
    }


    /** Generates a new scatter ray using BSDF importance sampling.
        \param[in] sd Shading data.
        \param[in] mi Material instance at the shading point.
        \param[in,out] path The path state.
        \return True if a ray was generated, false otherwise.
    */
    bool generateScatterRay(const BSDFSample result, const ShadingData sd, const IMaterialInstance mi, inout PathState path, bool valid)
    {
        if (valid) valid = generateScatterRay(result, sd, mi, path);

        // Ignore valid on purpose for now.
        if (kOutputNRDData)
        {
            const uint lobeTypes = mi.getLobeTypes(sd);
            const bool hasDeltaTransmissionLobe = (lobeTypes & (uint)LobeType::DeltaTransmission) != 0;
            const bool hasNonDeltaLobes = (lobeTypes & (uint)LobeType::NonDelta) != 0;

            if (path.getVertexIndex() == 1)
            {
                path.setDiffusePrimaryHit(result.isLobe(LobeType::Diffuse));
                path.setSpecularPrimaryHit(result.isLobe(LobeType::Specular));

                if (kOutputNRDAdditionalData)
                {
                    // Mark path as delta-only if it followed delta lobe on the primary hit, even though there might have been non-delta lobes.
                    path.setDeltaOnlyPath(result.isLobe(LobeType::DeltaReflection) || result.isLobe(LobeType::DeltaTransmission));

                    path.setDeltaReflectionPrimaryHit(result.isLobe(LobeType::DeltaReflection));
                    path.setDeltaTransmissionPath(result.isLobe(LobeType::DeltaTransmission));
                }
            }

            if (path.getVertexIndex() > 1)
            {
                if (hasNonDeltaLobes) path.setDeltaOnlyPath(false);

                if (kOutputNRDAdditionalData && path.isDeltaTransmissionPath() && path.isDeltaOnlyPath() && hasDeltaTransmissionLobe)
                {
                    if (result.isLobe(LobeType::DeltaReflection) && !isDeltaReflectionAllowedAlongDeltaTransmissionPath(sd))
                    {
                        path.setDeltaTransmissionPath(false);
                    }
                }
            }
        }

        return valid;
    }

    /** Generates a new scatter ray given a valid BSDF sample.
        \param[in] bs BSDF sample (assumed to be valid).
        \param[in] sd Shading data.
        \param[in] mi Material instance at the shading point.
        \param[in,out] path The path state.
        \return True if a ray was generated, false otherwise.
    */
    bool generateScatterRay(const BSDFSample bs, const ShadingData sd, const IMaterialInstance mi, inout PathState path)
    {
        const bool isTriangleHit = path.hit.getType() == HitType::Triangle;
        const bool isCurveHit = kUseCurves && (path.hit.getType() == HitType::Curve);
        const bool isCurveOTSHit = kUseCurves && (path.hit.getType() == HitType::CurveOTS);
        const bool isHairMaterial = kUseHairMaterial && (sd.mtl.getMaterialType() == MaterialType::Hair);
        // TODO: Decouple geometry from the material.
        const bool isCurvePolyTubeHit = isTriangleHit && isHairMaterial;

        path.dir = bs.wo;
        updatePathThroughput(path, bs.weight);

        if (kUseScreenSpaceReSTIR && ScreenSpaceReSTIR::kUseGI && path.getVertexIndex() == 1 && path.reSTIRGIData.isDisabled == false)
        {
            path.thp = 1.f / bs.pdf;
        }
        path.pdf = bs.pdf;

        path.clearEventFlags();

        // Handle reflection events.
        if (bs.isLobe(LobeType::Reflection))
        {
            // We classify specular events as diffuse if the roughness is above some threshold.
            float roughness = mi.getProperties(sd).roughness;
            bool isDiffuse = bs.isLobe(LobeType::DiffuseReflection) || roughness > params.specularRoughnessThreshold;
            if (isDiffuse)
            {
                path.incrementBounces(BounceType::Diffuse);
            }
            else
            {
                path.incrementBounces(BounceType::Specular);
                path.setSpecular();
            }
        }

        // Handle delta events.
        if (bs.isLobe(LobeType::Delta))
        {
            path.setDelta();
        }

        // Handle transmission events.
        if (bs.isLobe(LobeType::Transmission))
        {
            path.incrementBounces(BounceType::Transmission);
            path.setTransmission();

            if (isCurveHit)
            {
                // No need to offset the origin in this case.
            }
            else if (isCurveOTSHit)
            {
                // For curves tessellated into orthogonal triangle strips, we make sure that the origin is on the same side as a scatter ray direction
                // so there is no self-intersection.
                const float otsWidthScale = 1.11f;
                path.origin = sd.posW - otsWidthScale * sd.faceN * sd.curveRadius;
            }
            else if (isCurvePolyTubeHit)
            {
                // For curves tessellated into poly-tubes, we make sure that the origin is on the same side as a scatter ray direction
                // so there is no self-intersection.
                path.origin = sd.posW - sd.frame.N * sd.curveRadius * 2.1f;
            }
            else
            {
                // Compute ray origin for next ray segment.
                path.origin = sd.computeRayOrigin(false);

                // Update interior list and inside volume flag.
                if (!sd.mtl.isThinSurface())
                {
                    uint nestedPriority = sd.mtl.getNestedPriority();
                    path.interiorList.handleIntersection(sd.materialID, nestedPriority, sd.frontFacing);
                    path.setInsideDielectricVolume(!path.interiorList.isEmpty());

                    if (kSupportPathStateVolumeProperties)
                        setVolumePropertiesOnPath( mi, sd.frontFacing, bs.lobeType, path);
                }
            }
        }

        // Save the normal used for NEE. This is needed for MIS.
        path.normal = sd.getOrientedFaceNormal();

        // Mark the path as valid only if it has a non-zero throughput.
        bool valid = any(path.thp > 0.f);

        return valid;
    }

    /** Generates a new scatter ray using phase function importance sampling.
        \param[in,out] path The path state.
        \return True if a ray was generated, false otherwise.
    */
    bool generateVolumeScatterRay(inout PathState path, const IPhaseFunction pf)
    {
        path.clearEventFlags();

        float3 weight;
        bool valid = pf.sample(-path.dir, path.dir, path.pdf, weight, path.sg);
        if (valid)
        {
            updatePathThroughput(path, weight);
            path.incrementBounces(BounceType::Volume);
            path.setVolume();
        }

        return valid;
    }

    /** Evaluates the currently configured heuristic for multiple importance sampling (MIS).
        \param[in] n0 Number of samples taken from the first sampling strategy.
        \param[in] p0 Pdf for the first sampling strategy.
        \param[in] n1 Number of samples taken from the second sampling strategy.
        \param[in] p1 Pdf for the second sampling strategy.
        \return Weight for the contribution from the first strategy (p0).
    */
    float evalMIS(float n0, float p0, float n1, float p1)
    {
        switch (MISHeuristic(kMISHeuristic))
        {
        case MISHeuristic::Balance:
        {
            // Balance heuristic
            float q0 = n0 * p0;
            float q1 = n1 * p1;
            return q0 / (q0 + q1);
        }
        case MISHeuristic::PowerTwo:
        {
            // Power two heuristic
            float q0 = (n0 * p0) * (n0 * p0);
            float q1 = (n1 * p1) * (n1 * p1);
            return q0 / (q0 + q1);
        }
        case MISHeuristic::PowerExp:
        {
            // Power exp heuristic
            float q0 = pow(n0 * p0, kMISPowerExponent);
            float q1 = pow(n1 * p1, kMISPowerExponent);
            return q0 / (q0 + q1);
        }
        default:
            return 0.f;
        }
    }

    /** Generates a light sample on the environment map.
        \param[in] vertex Path vertex.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateEnvMapSample(const PathVertex vertex, inout SampleGenerator sg, out LightSample ls)
    {
        ls = {}; // Default initialization to avoid divergence at returns.

        if (!kUseEnvLight) return false;

        // Sample environment map.
        EnvMapSample lightSample;
        if (!envMapSampler.sample(sampleNext2D(sg), lightSample)) return false;

        // Setup returned sample.
        ls.Li = lightSample.pdf > 0.f ? lightSample.Le / lightSample.pdf : float3(0);
        ls.pdf = lightSample.pdf;
        ls.origin = vertex.getRayOrigin(lightSample.dir);
        ls.distance = kRayTMax;
        ls.dir = lightSample.dir;

        return any(ls.Li > 0.f);
    }

    /** Generates a light sample on the emissive geometry.
        \param[in] vertex Path vertex.
        \param[in] upperHemisphere True if only upper hemisphere should be considered.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateEmissiveSample(const PathVertex vertex, const bool upperHemisphere, inout SampleGenerator sg, out LightSample ls)
    {
        ls = {}; // Default initialization to avoid divergence at returns.
        if (!kUseEmissiveLights) return false;

        TriangleLightSample tls;
        if (!emissiveSampler.sampleLight(vertex.pos, vertex.getOrientedFaceNormal(), upperHemisphere, sg, tls)) return false;

        // Setup returned sample.
        ls.Li = tls.pdf > 0.f ? tls.Le / tls.pdf : float3(0);
        ls.pdf = tls.pdf;
        // Offset shading and light position to avoid self-intersection.
        float3 lightPos = computeRayOrigin(tls.posW, tls.normalW);
        ls.origin = vertex.getRayOrigin(lightPos - vertex.pos);
        float3 toLight = lightPos - ls.origin;
        ls.distance = length(toLight);
        ls.dir = normalize(toLight);

        return any(ls.Li > 0.f);
    }

    /** Generates a light sample on the analytic lights.
        \param[in] vertex Path vertex.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateAnalyticLightSample(const PathVertex vertex, inout SampleGenerator sg, out LightSample ls)
    {
        ls = {}; // Default initialization to avoid divergence at returns.

        uint lightCount = gScene.getLightCount();
        if (!kUseAnalyticLights || lightCount == 0) return false;

        // Sample analytic light source selected uniformly from the light list.
        // TODO: Sample based on estimated contributions as pdf.
        uint lightIndex = min(uint(sampleNext1D(sg) * lightCount), lightCount - 1);

        // Sample local light source.
        AnalyticLightSample lightSample;
        if (!sampleLight(vertex.pos, gScene.getLight(lightIndex), sg, lightSample)) return false;

        // Setup returned sample.
        ls.pdf = lightSample.pdf / lightCount;
        ls.Li = lightSample.Li * lightCount;
        // Offset shading position to avoid self-intersection.
        ls.origin = vertex.getRayOrigin(lightSample.dir);
        // Analytic lights do not currently have a geometric representation in the scene.
        // Do not worry about adjusting the ray length to avoid self-intersections at the light.
        ls.distance = lightSample.distance;
        ls.dir = lightSample.dir;

        return any(ls.Li > 0.f);
    }

    /** Return the probabilities for selecting different light types.
        \param[out] p Probabilities.
    */
    void getLightTypeSelectionProbabilities(out float p[3])
    {
        // Set relative probabilities of the different sampling techniques.
        // TODO: These should use estimated irradiance from each light type. Using equal probabilities for now.
        p[0] = kUseEnvLight ? 1.f : 0.f;
        p[1] = kUseEmissiveLights ? 1.f : 0.f;
        p[2] = kUseAnalyticLights ? 1.f : 0.f;

        // Normalize probabilities. Early out if zero.
        float sum = p[0] + p[1] + p[2];
        if (sum == 0.f) return;

        float invSum = 1.f / sum;
        p[0] *= invSum;
        p[1] *= invSum;
        p[2] *= invSum;
    }

    float getEnvMapSelectionProbability()   { float p[3]; getLightTypeSelectionProbabilities(p); return p[0]; }
    float getEmissiveSelectionProbability() { float p[3]; getLightTypeSelectionProbabilities(p); return p[1]; }
    float getAnalyicSelectionProbability()  { float p[3]; getLightTypeSelectionProbabilities(p); return p[2]; }

    /** Select a light type for sampling.
        \param[out] lightType Selected light type.
        \param[out] pdf Probability for selected type.
        \param[in,out] sg Sample generator.
        \return Return true if selection is valid.
    */
    bool selectLightType(out uint lightType, out float pdf, inout SampleGenerator sg)
    {
        float p[3];
        getLightTypeSelectionProbabilities(p);

        float u = sampleNext1D(sg);

        [unroll]
        for (lightType = 0; lightType < 3; ++lightType)
        {
            if (u < p[lightType])
            {
                pdf = p[lightType];
                return true;
            }
            u -= p[lightType];
        }

        lightType = {};
        pdf = {};

        return false;
    }

    /** Samples a light source in the scene.
        This function first stochastically selects a type of light source to sample,
        and then calls that the sampling function for the chosen light type.
        \param[in] vertex Path vertex.
        \param[in] sampleUpperHemisphere True if the upper hemisphere should be sampled.
        \param[in] sampleLowerHemisphere True if the lower hemisphere should be sampled.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateLightSample(const PathVertex vertex, const bool sampleUpperHemisphere, const bool sampleLowerHemisphere, inout SampleGenerator sg, out LightSample ls)
    {
        ls = {};

        uint lightType;
        float selectionPdf;
        if (!selectLightType(lightType, selectionPdf, sg)) return false;

        bool valid = false;
        if (kUseEnvLight && lightType == (uint)LightType::EnvMap) valid = generateEnvMapSample(vertex, sg, ls);
        if (kUseEmissiveLights && lightType == (uint)LightType::Emissive)
        {
            // Emissive light samplers have an option to exclusively sample the upper hemisphere.
            bool upperHemisphere = sampleUpperHemisphere && !sampleLowerHemisphere;
            valid = generateEmissiveSample(vertex, upperHemisphere, sg, ls);
        }
        if (kUseAnalyticLights && lightType == (uint)LightType::Analytic)
        {
            valid = generateAnalyticLightSample(vertex, sg, ls);
        }
        if (!valid) return false;

        // Reject samples in non-requested hemispheres.
        float NdotL = dot(vertex.getOrientedFaceNormal(), ls.dir);
        if ((!sampleUpperHemisphere && NdotL >= -kMinCosTheta) || (!sampleLowerHemisphere && NdotL <= kMinCosTheta))
            return false;

        // Account for light type selection.
        ls.lightType = lightType;
        ls.pdf *= selectionPdf;
        ls.Li /= selectionPdf;

        return true;
    }

    enum NextHitEvent
    {
        NoVolumeScattering,             ///< No volume scattering.
        HomogeneousVolumeScattering,    ///< Homogeneous volume scattering.
        GridVolumeScattering,           ///< Grid volume scattering.
    };

    /** Finds the next surface or volume hit.
        If there are volumes in the scene, this function first samples a free flight distance.
        Then a (shortened) ray is traced against the scene to find any surface hits.
        This is done using the provided IClosestHitQuery adaptor to allow using different tracing APIs (TraceRay, TracerRayInline).
        Finally, using both the free flight distance and the closest surface hit, the final hit is determined and stored in the path state.
        \param[in,out] path The path state.
        \param[in,out] chq Closest hit query.
    */
    void nextHit<ClosestHitQuery : IClosestHitQuery>(inout PathState path, inout ClosestHitQuery chq)
    {
        if (kUseScreenSpaceReSTIR && ScreenSpaceReSTIR::kUseGI && path.getVertexIndex() == 1)
        {
            path.reSTIRGIData.L0 = path.L;
        }

        NextHitEvent event = NextHitEvent::NoVolumeScattering;

        HomogeneousVolumeSampler::DistanceSample hds;   ///< Distance sample in homogeneous volume.
        GridVolumeSampler::DistanceSample gds;          ///< Distance sample in grid volume.

        Ray ray = path.getScatterRay();

        // Check for scattering in both homogeneous and grid volumes.
        // Homogeneous volumes have precendence.
        if (!path.interiorList.isEmpty())
        {
            const uint materialID = path.interiorList.getTopMaterialID();
            let material = gScene.materials.getMaterial(materialID);
            VolumeProperties vp = material.getHomogeneousVolumeProperties(gScene.materials, materialID);

            // take sigmaA/S from the path instead of the volume
            if (kSupportPathStateVolumeProperties && path.hasVolumeProperties())
            {
                vp.sigmaA = path.sigmaA;
                vp.sigmaS = path.sigmaS;
            }

            // Sample free flight distance only if we need to sample a scattering event.
            // Transmittance through the last path segment is evaluated separately in `finalizeNextHitQuery`.
            if (kMaxVolumeBounces > 0 && path.getBounces(BounceType::Volume) < kMaxVolumeBounces && any(vp.sigmaS > 0.f))
            {
                if (HomogeneousVolumeSampler::sampleDistance(vp, path.thp, path.sg, hds))
                {
                    ray.tMax = hds.t;
                    event = NextHitEvent::HomogeneousVolumeScattering;
                }
            }
        }
        else if (kUseGridVolumes)
        {
            // Sample free flight distance even if we are on the last path segment.
            // This is required because currently transmittance for the last path segment is estimated using free flight sampling.
            // TODO: Consider switching to using transmittance estimators for last path segment.
            if (path.getBounces(BounceType::Volume) <= kMaxVolumeBounces)
            {
                if (gridVolumeSampler.sampleDistance(ray, path.sg, gds))
                {
                    ray.tMax = gds.t;
                    event = NextHitEvent::GridVolumeScattering;
                }
            }
        }

        // Advance to next path vertex.
        path.incrementVertexIndex();

        // Trace ray.
        logTraceRay(PixelStatsRayType::ClosestHit);
        HitInfo hit;
        float hitT;
        if (!chq.traceRay(path, ray, hit, hitT)) hit = {};

        switch (event)
        {
        case NextHitEvent::NoVolumeScattering:
            {
                if (hit.isValid())
                {
                    path.setHit(hit);

                    if (!path.interiorList.isEmpty())
                    {
                        const uint materialID = path.interiorList.getTopMaterialID();
                        let material = gScene.materials.getMaterial(materialID);
                        VolumeProperties vp = material.getHomogeneousVolumeProperties(gScene.materials, materialID);

                        // take sigmaA/S from the path instead of the volume
                        if (kSupportPathStateVolumeProperties && path.hasVolumeProperties())
                        {
                            vp.sigmaA = path.sigmaA;
                            vp.sigmaS = path.sigmaS;
                        }

                        updatePathThroughput(path, HomogeneousVolumeSampler::evalTransmittance(vp, hitT));
                    }
                }
                else
                {
                    path.clearHit();
                }
            }
            break;
        case NextHitEvent::HomogeneousVolumeScattering:
            {
                const bool hasScattered = !hit.isValid();
                updatePathThroughput(path, hds.evalThroughput(hasScattered, hitT));
                if (hasScattered)
                {
                    hitT = hds.t;
                    VolumeHit volumeHit;
                    volumeHit.t = hitT;
                    path.setHit(HitInfo(volumeHit));
                }
                else
                {
                    path.setHit(hit);
                }
            }
            break;
        case NextHitEvent::GridVolumeScattering:
            if (kUseGridVolumes)
            {
                const bool hasScattered = !hit.isValid();
                if (hasScattered)
                {
                    hitT = gds.t;
                    updatePathThroughput(path, gds.thp);
                    VolumeHit volumeHit;
                    volumeHit.t = hitT;
                    volumeHit.g = gScene.getGridVolume(0).getAnisotropy();
                    path.setHit(HitInfo(volumeHit));
                }
                else
                {
                    path.setHit(hit);
                }
            }
            break;
        }

        // Accumulate path length.
        if (kOutputNRDData || kOutputGuideData)
        {
            if (path.isHit())
            {
                path.sceneLength += float16_t(hitT);
            }
            else
            {
                path.sceneLength = float16_t(kNRDInvalidPathLength);
            }
        }
    }

    /** Called for the last path vertex on a surface.
        This allows for querying a radiance cache at the end of a path.
        \param[in,out] path Path state.
        \param[in] thp Path throughput.
        \param[in] sd Shading data.
    */
    void handleLastSurfaceVertex(inout PathState path, const float3 thp, const ShadingData sd, const BSDFProperties bsdfProperties)
    {
    }

    /** Handle hits on dielectrics.
        \return True if this is an valid intersection, false if it is rejected.
    */
    bool handleNestedDielectrics(inout ShadingData sd, inout PathState path)
    {
        // Check for false intersections.
        uint nestedPriority = sd.mtl.getNestedPriority();
        if (!path.interiorList.isTrueIntersection(nestedPriority))
        {
            // If it is a false intersection, we reject the hit and continue the path
            // on the other side of the interface.
            // If the offset position is not quite large enough, due to self-intersections
            // it is possible we repeatedly hit the same surface and try to reject it.
            // This has happened in a few occasions with large transmissive triangles.
            // As a workaround, count number of rejected hits and terminate the path if too many.
            if (path.rejectedHits < kMaxRejectedHits)
            {
                path.rejectedHits++;
                path.interiorList.handleIntersection(sd.materialID, nestedPriority, sd.frontFacing);
                path.origin = sd.computeRayOrigin(false);
                path.decrementVertexIndex();
            }
            else
            {
                path.terminate();
            }
            return false;
        }

        // Compute index of refraction for medium on the outside.
        sd.IoR = computeOutsideIoR(path.interiorList, sd.materialID, sd.frontFacing);

        // Limit specular transmission paths.
        if (kLimitTransmission)
        {
            const uint transmissionBounces = path.getBounces(BounceType::Transmission);
            if (transmissionBounces >= kMaxTransmissionReflectionDepth) sd.mtl.setActiveLobes((uint)LobeType::All & ~((uint)LobeType::DeltaReflection | (uint)LobeType::SpecularReflection));
            if (transmissionBounces >= kMaxTransmissionRefractionDepth) sd.IoR = gScene.materials.evalIoR(sd.materialID); // Set outside IoR same as IoR inside material at hit. This will disable refraction.
        }

        return true;
    }

    /** Apply russian roulette to terminate paths early.
        \param[in,out] path Path.
        \param[in] u Uniform random number in [0,1).
        \return Returns true if path was terminated.
    */
    bool terminatePathByRussianRoulette(inout PathState path, float u)
    {
        const float rrVal = luminance(path.thp);
        const float prob = max(0.f, 1.f - rrVal);
        if (u < prob)
        {
            if (kUseNRC)
            {
                path.thp = 0;
                path.nrcData.trainingVertexThroughput = 0;
            }
            path.terminate();
            return true;
        }
        path.thp /= 1.f - prob;
        if (kUseNRC) path.nrcData.trainingVertexThroughput /= 1.f - prob;
        return false;
    }

    /** Handle the case when a scatter ray hits the scene (surface or volume).
        After handling the hit, a new scatter ray is generated or the path is terminated.
        \param[in,out] path The path state.
        \param[in,out] vq Visibility query.
    */
    void handleHit<VisibilityQuery : IVisibilityQuery>(inout PathState path, inout VisibilityQuery vq)
    {
        if (path.hit.getType() == HitType::Volume)
        {
            handleVolumeHit(path, vq);
        }
        else
        {
            handleSurfaceHit(path, vq);
        }
    }

    /** Helper to create a texture sampler instance.
        The method for computing texture level-of-detail depends on the configuration.
        \param[in] path Path state.
        \param[in] isPrimaryTriangleHit True if primary hit on a triangle.
        \return Texture sampler instance.
    */
    ITextureSampler createTextureSampler(inout PathState path, bool isPrimaryHit, bool isTriangleHit)
    {
        if (kPrimaryLodMode == TexLODMode::RayDiffs && isPrimaryHit && isTriangleHit)
        {
            // Filtered lookups at primary hit on triangle.
            float2 ddx, ddy;
            computeDerivativesAtPrimaryTriangleHit(path.hit.getTriangleHit(), path.getPixel(), params.frameDim, ddx, ddy);
            return ExplicitGradientTextureSampler(ddx, ddy);
        }
        else if (kPrimaryLodMode == TexLODMode::RayCones && isPrimaryHit && isTriangleHit)
        {
            // Compute hit point, direction and distance
            const TriangleHit triangleHit = path.hit.getTriangleHit();
            StaticVertexData vertices[3];
            VertexData v = gScene.getVertexData(triangleHit, vertices);
            float3 dir = normalize(v.posW - path.origin);
            float hitT = length(path.origin - v.posW);

            // Create ray cone at camera, then propagate to hit point
            RayCone rayCone = RayCone(0.f, params.pixelSpreadAngle);
            rayCone = rayCone.propagateDistance(hitT);

            // Compute texture LOD.
            float triLODConstant = computeRayConeTriangleLODValue(vertices, float3x3(gScene.getWorldMatrix(triangleHit.instanceID)));
            float lambda = rayCone.computeLOD(triLODConstant, path.dir, v.faceNormalW);

            return ExplicitRayConesLodTextureSampler(lambda);
        }
        else if (kPrimaryLodMode == TexLODMode::Stochastic && isPrimaryHit && isTriangleHit)
        {
            let xy = path.getPixel();
            let stbnOffset0 = int2(7, 11);

            // Filtered lookups at primary hit on triangle.
            float2 ddx, ddy;
            computeDerivativesAtPrimaryTriangleHit(path.hit.getTriangleHit(), path.getPixel(), params.frameDim, ddx, ddy);

            float2 uvJitter = 0;
            if (kUseSpatioTemporalBlueNoise)
                uvJitter += float2(spatioTemporalBlueNoise[xy % 128],
                                   spatioTemporalBlueNoise[(xy + stbnOffset0) % 128]);
            else
                uvJitter += sampleNext2D(path.sg);

            if (kUseStochasticBicubicFiltering)
            {
                if (kUseSpatioTemporalBlueNoise)
                {
                    let stbnOffset1u = int2(13, 17);
                    let stbnOffset2u = int2(19, 21);
                    let stbnOffset1v = stbnOffset1u + stbnOffset0;
                    let stbnOffset2v = stbnOffset2u + stbnOffset0;

                    uvJitter += float2(spatioTemporalBlueNoise[(xy + stbnOffset1u) % 128],
                                       spatioTemporalBlueNoise[(xy + stbnOffset1v) % 128]);

                    uvJitter += float2(spatioTemporalBlueNoise[(xy + stbnOffset2u) % 128],
                                       spatioTemporalBlueNoise[(xy + stbnOffset2v) % 128]);
                }
                else
                {
                    uvJitter += sampleNext2D(path.sg) + sampleNext2D(path.sg);
                }
            }

            // Add fine stochastic noise to coarse blue noise
            if (kUseSpatioTemporalBlueNoise)
                uvJitter += sampleNext2D(path.sg) / 256.0;

            // Zero-mean distribution
            uvJitter -= kUseStochasticBicubicFiltering ? 1.5f : 0.5f;

            let lodJitter = sampleNext1D(path.sg) - 0.5f;
            return StochasticTextureSampler(ddx, ddy, uvJitter, lodJitter);
        }
        else
        {
            float lod = isPrimaryHit ? 0.f : params.lodBias;
            return ExplicitLodTextureSampler(lod);
        }
    }

    /** Handle the case when a scatter ray hits a surface.
        After handling the hit, a new scatter ray is generated or the path is terminated.
        \param[in,out] path The path state.
        \param[in,out] vq Visibility query.
    */
    void handleSurfaceHit<VisibilityQuery : IVisibilityQuery>(inout PathState path, inout VisibilityQuery vq)
    {
        // Upon hit:
        // - Load vertex/material data
        // - Compute MIS weight if path.getVertexIndex() > 1 and emissive hit
        // - Add emitted radiance
        // - Sample light(s) using shadow rays
        // - Sample scatter ray or terminate

        const bool isPrimaryHit = path.getVertexIndex() == 1;
        const HitType hitType = path.hit.getType();
        const bool isTriangleHit = hitType == HitType::Triangle;
        const bool isCurveHit = kUseCurves && (hitType == HitType::Curve);
        const bool isCurveOTSHit = kUseCurves && (hitType == HitType::CurveOTS);

        let lod = createTextureSampler(path, isPrimaryHit, isTriangleHit);

        // Load shading data. This is a long latency operation.
        ShadingData sd = loadShadingData(path.hit, path.origin, path.dir, lod);

        if (kOutputGuideData && path.getVertexIndex() == 2) path.guideData.specHitDist = path.isSpecular() ? path.sceneLength : 0.f;

        const bool isHairMaterial = kUseHairMaterial && (sd.mtl.getMaterialType() == MaterialType::Hair);
        // TODO: Decouple geometry from the material.
        const bool isCurvePolyTubeHit = isTriangleHit && isHairMaterial;

        // Draw a random number for terminating paths using russian roulette before drawing any other random numbers.
        // This allows predicting termination in getCoherenceHints(), which is useful for reordering schedulers.
        float uRussianRoulette = 0.f;
        if (kUseRussianRoulette) uRussianRoulette = sampleNext1D(path.sg);

        // Reject false hits in nested dielectrics.
        if (!handleNestedDielectrics(sd, path)) return;

        logPathVertex();

        // Create material instance and query its properties.
        // TODO: Using `let mi = ..` causes a compiler crash. Try again after https://gitlab-master.nvidia.com/slang/slang/-/issues/114 is resolved.
        let hints = getMaterialInstanceHints(path.hit, isPrimaryHit);
        const IMaterialInstance mi = gScene.materials.getMaterialInstance(sd, lod, hints);
        BSDFProperties bsdfProperties = mi.getProperties(sd);
        warpProfile(WarpProfilerEvent::MaterialCreate, path.getVertexIndex(), sd.materialID);

        // DEMO21: Modify the emission on geometry with applied IES profile
        evaluateLightProfile(bsdfProperties, sd, path);

        if (kUseScreenSpaceReSTIR && ScreenSpaceReSTIR::kUseGI)
        {
            if (isPrimaryHit)
            {
                // ReSTIR uses a simple material model with only diffuse and specular reflection lobes.
                // We query the BSDF for the diffuse albedo and specular reflectance, and use their luminances as weights.
                // TODO: https://gitlab-master.nvidia.com/nvresearch-gfx/Tools/Falcor/-/issues/1406
                path.reSTIRGIData.creationPoint = sd.posW;
                path.reSTIRGIData.creationNormal = bsdfProperties.guideNormal;
                float roughness = max(0.08f, bsdfProperties.roughness); // TODO: Remove/move clamp into ReSTIR.
                float ggxAlpha = roughness * roughness;
                if (bsdfProperties.isTransmissive ||
                    ScreenSpaceReSTIR::ignoreReSTIRGI(ggxAlpha, luminance(bsdfProperties.diffuseReflectionAlbedo), luminance(bsdfProperties.specularReflectance)))
                {
                    path.reSTIRGIData.isDisabled = true;
                }
            }
            else if (path.getVertexIndex() == 2)
            {
                path.reSTIRGIData.position = sd.posW;
                path.reSTIRGIData.normal = bsdfProperties.guideNormal;
            }
        }

        // Disable specular lobes if caustics are disabled and path already contains a diffuse vertex.
        bool isSpecular = bsdfProperties.roughness <= params.specularRoughnessThreshold;
        if (kDisableCaustics && path.getBounces(BounceType::Diffuse) > 0 && isSpecular)
        {
            sd.mtl.setActiveLobes((uint)LobeType::Diffuse);
        }

        // Optionally disable emission inside volumes.
        if (!kUseLightsInDielectricVolumes && path.isInsideDielectricVolume())
        {
            bsdfProperties.emission = float3(0.f);
        }

        // Check if the scatter event is samplable by the light sampling technique.
        const bool isLightSamplable = path.isLightSamplable();

        // Add emitted radiance.
        // The primary hit is always included, secondary hits only if emissive lights are enabled and the full light contribution hasn't been sampled elsewhere.
        bool computeEmissive = isPrimaryHit || kUseEmissiveLights && (!kUseNEE || kUseMIS || !path.isLightSampled() || !isLightSamplable);

        // Disable all contributions to direct illumination if requested.
        if (kDisableDirectIllumination && path.getVertexIndex() == 2) computeEmissive = false;

        // With screen space ReSTIR enabled, we sample the full direct illumination
        // contribution on the primary hit. Skip any additional contribution on the
        // secondary hit unless it comes from a scatter event that ReSTIR cannot handle,
        // such as transmission, delta or volume scattering events.
        if (kUseScreenSpaceReSTIR && ScreenSpaceReSTIR::kUseDI && path.getVertexIndex() == 2 && !path.isTransmission() && !path.isDelta() && !path.isVolume()) computeEmissive = false;

        float3 attenuatedEmission = 0.f;

        if (computeEmissive && any(bsdfProperties.emission > 0.f))
        {
            float misWeight = 1.f;
            if (kUseEmissiveLights && kUseNEE && kUseMIS && isTriangleHit && !isPrimaryHit && path.isLightSampled() && isLightSamplable)
            {
                // If NEE and MIS are enabled, and we've already sampled emissive lights,
                // then we need to evaluate the MIS weight here to account for the remaining contribution.
                // Note that MIS is only applied for hits on emissive triangles (other emissive geometry is not supported).

                // Prepare hit point struct with data needed for emissive light PDF evaluation.
                TriangleHit triangleHit = path.hit.getTriangleHit();
                TriangleLightHit hit;
                hit.triangleIndex = gScene.lightCollection.getTriangleIndex(triangleHit.instanceID, triangleHit.primitiveIndex);
                hit.posW = sd.posW;
                hit.normalW = sd.getOrientedFaceNormal();

                // Evaluate PDF at the hit, had it been generated with light sampling.
                // Emissive light samplers have an option to exclusively sample the upper hemisphere.
                bool upperHemisphere = path.isLightSampledUpper() && !path.isLightSampledLower();
                float lightPdf = getEmissiveSelectionProbability() * emissiveSampler.evalPdf(path.origin, path.normal, upperHemisphere, hit);

                // Compute MIS weight by combining this with BSDF sampling.
                // Note we can assume path.pdf > 0.f since we shouldn't have got here otherwise.
                misWeight = evalMIS(1, path.pdf, 1, lightPdf);
            }

            // Accumulate emitted radiance weighted by path throughput and MIS weight.
            addToPathContribution(path, misWeight * bsdfProperties.emission);

            attenuatedEmission = path.thp * misWeight * bsdfProperties.emission;
        }

        // Terminate after scatter ray on last vertex has been processed.
        bool lastVertex = hasFinishedSurfaceBounces(path);
        bool lastVertexNRC = kUseNRC && !path.nrcData.isForTraining() && path.nrcData.isQueryVertex();
        if (lastVertex || lastVertexNRC)
        {
            path.terminate();
            return;
        }

        // Update the path spread approximation, used to trigger the path termination heuristic.
        // The heuristic prevents querying the Neural Radiance Cache before the signal has been sufficiently
        // blurred by the path spread.
        if (kUseNRC)
        {
            if (path.isVolume())
            {
                // TODO: update this to grid or homog vol distance sample
                const VolumeHit volumeHit = path.hit.getVolumeHit();
                path.nrcData.updatePathSpread(path.getVertexIndex(), volumeHit.t, -path.dir, -path.dir, path.pdf, path.isDelta());
            }
            else
            {
                float distance = sqrt(dot(sd.posW - path.origin, sd.posW - path.origin));
                path.nrcData.updatePathSpread(path.getVertexIndex(), distance, sd.V, sd.frame.N, path.pdf, path.isDelta());
            }
        }

        // Compute origin for rays traced from this path vertex.
        if (isCurveHit)
        {
            // For curves, we set the new origin at the sphere center.
            path.origin = sd.posW - sd.frame.N * sd.curveRadius;
        }
        else if (isCurveOTSHit)
        {
            // For curves tessellated into orthogonal triangle strips, we offset the new origin away from the curve center.
            const float otsWidthScale = 1.11f;
            path.origin = sd.posW + otsWidthScale * sd.faceN * sd.curveRadius;
        }
        else if (isCurvePolyTubeHit)
        {
            // For curves tessellated into poly-tubes, we offset the new origin away from the curve center.
            path.origin = sd.posW + sd.frame.N * sd.curveRadius * 0.1f;
        }
        else
        {
            path.origin = sd.computeRayOrigin();
        }

        // Generate a BSDF sample already here if NRC is enabled.
        // In order to decide whether to use NRC on this vertex or a later one (e.g. when the sampled reflection model is a delta function),
        // the BSDF sample must be known before calling nrc.createQuery(), which implements the vertex skipping routine.
        bool validBSDFSample = true;
        BSDFSample bsdfSample;
        if (kUseNRC)
        {
            if (!path.isVolume())
            {
                validBSDFSample = mi.sample(sd, path.sg, bsdfSample, kUseBSDFSampling);
                warpProfile(WarpProfilerEvent::MaterialSample, path.getVertexIndex(), sd.materialID);
            }

            if (nrc.createQuery(path, sd, mi.getProperties(sd),validBSDFSample && bsdfSample.isLobe(LobeType::Delta), path.isVolume()))
            {
                path.terminate();
                return;
            }
        }


        // Determine if material instance has non-delta lobes.
        const uint lobeTypes = mi.getLobeTypes(sd);
        const bool hasNonDeltaLobes = (lobeTypes & (uint)LobeType::NonDelta) != 0;

        // Check if we should apply NEE.
        const bool applyNEE = kUseNEE && hasNonDeltaLobes;

        // Check if sample from screen space ReSTIR should be applied instead of NEE.
        const bool applyScreenSpaceReSTIR = kUseScreenSpaceReSTIR && ScreenSpaceReSTIR::kUseDI && isPrimaryHit && hasNonDeltaLobes;

        // Check if lights should be sampled.
        bool sampleLights = (applyNEE || applyScreenSpaceReSTIR);

        // Disable all constributions to direct illumination if requested.
        if (kDisableDirectIllumination && isPrimaryHit) sampleLights = false;

        // TODO: Support multiple shadow rays.
        path.setLightSampled(false, false);
        if (sampleLights)
        {
            LightSample ls = {};
            bool validSample = false;
            if (applyScreenSpaceReSTIR)
            {
                validSample = screenSpaceReSTIR.getFinalSample(path.getPixel(), ls.dir, ls.distance, ls.Li);
                ls.origin = path.origin;
            }
            else
            {
                // Setup path vertex.
                PathVertex vertex = PathVertex(path.getVertexIndex(), sd.posW, sd.faceN, sd.frontFacing);

                // Determine if upper/lower hemispheres need to be sampled.
                bool sampleUpperHemisphere = isCurveHit || isCurvePolyTubeHit || isCurveOTSHit || ((lobeTypes & (uint)LobeType::NonDeltaReflection) != 0);
                if (!kUseLightsInDielectricVolumes && path.isInsideDielectricVolume()) sampleUpperHemisphere = false;
                bool sampleLowerHemisphere = isCurveHit || isCurvePolyTubeHit || isCurveOTSHit || ((lobeTypes & (uint)LobeType::NonDeltaTransmission) != 0);

                // Sample a light.
                validSample = generateLightSample(vertex, sampleUpperHemisphere, sampleLowerHemisphere, path.sg, ls);
                path.setLightSampled(sampleUpperHemisphere, sampleLowerHemisphere);
            }

            if (validSample)
            {
                // Apply MIS weight.
                if (kUseMIS && !applyScreenSpaceReSTIR && ls.lightType != (uint)LightType::Analytic)
                {
                    float scatterPdf = mi.evalPdf(sd, ls.dir, kUseBSDFSampling);
                    warpProfile(WarpProfilerEvent::MaterialEvalPdf, path.getVertexIndex(), sd.materialID);
                    ls.Li *= evalMIS(1, ls.pdf, 1, scatterPdf);
                }

                float3 weight = mi.eval(sd, ls.dir, path.sg);
                warpProfile(WarpProfilerEvent::MaterialEval, path.getVertexIndex(), sd.materialID);
                float3 Lr = weight * ls.Li;
                if (any(Lr > 0.f))
                {
                    Ray ray = ls.getVisibilityRay();
                    bool visible = true;

                    if (isCurveOTSHit)
                    {
                        // For curves tessellated into orthogonal triangle strips, we make sure that the origin is on the same side as light
                        // so there is no self-shadowing (transmission lobe of hair BSDF takes care of that).
                        if (dot(sd.faceN, ray.dir) < 0.f)
                        {
                            const float otsWidthScale = 1.11f;
                            ray.origin = sd.posW - otsWidthScale * sd.faceN * sd.curveRadius;
                        }
                    }
                    else if (isCurvePolyTubeHit)
                    {
                        // For curves tessellated into poly-tubes, we make sure that the origin is on the same side as light
                        // so there is no self-shadowing (transmission lobe of hair BSDF takes care of that).
                        if (dot(sd.frame.N, ray.dir) < 0.f)
                        {
                            ray.origin = ray.origin - sd.frame.N * sd.curveRadius * 2.1f;
                        }
                    }

                    // TODO: Checking `applyNEE` here is a WAR for a driver issue in ray query lowering,
                    // it shouldn't be necessary because this code is only executed if `applyNEE || applyScreenSpaceReSTIR`.
                    // https://nvbugswb.nvidia.com/NvBugs5/SWBug.aspx?bugid=3412536
                    if (applyNEE && !applyScreenSpaceReSTIR)
                    {
                        logTraceRay(PixelStatsRayType::Visibility);
                        visible = vq.traceVisibilityRay(ray);
                    }

                    if (visible)
                    {
                        if (kUseGridVolumes) Lr *= gridVolumeSampler.evalTransmittance(ray, path.sg);
                        addToPathContribution(path, Lr);
                    }
                }
            }
        }

        // Russian roulette to terminate paths early.
        if (kUseRussianRoulette)
        {
            // NRC imposes additional conditions when russian roulette is permissible.
            // TODO: check that these extra conditions are taken into account when constructing NEE probability for MIS
            if(!kUseNRC || (path.getVertexIndex() >= 3 && !path.nrcData.isUnbiased()))
            {
                if (terminatePathByRussianRoulette(path, uRussianRoulette))
                {
                    return;
                }
            }
        }

        // Only used if kOutputNRDData == true.
        const bool wasDeltaOnlyPathBeforeScattering = path.isDeltaOnlyPath();

        // Temporary keep path throughput from before the scatter event.
        float3 prevThp = path.thp;

        // Generate the next path segment or terminate.
        bool valid = false;
        if (kUseNRC)
        {
            if (validBSDFSample) valid = generateScatterRay(bsdfSample, sd, mi, path, validBSDFSample);
        }
        else
        {
            valid = generateScatterRay(sd, mi, path);
        }

        // Output guide data.
        if (path.getVertexIndex() == 1)
        {
            setPrimarySurfaceGuideData(path.guideData, sd, bsdfProperties);
        }
        if (path.getVertexIndex() == 2 && (path.getBounces(BounceType::Specular) == 1 || path.getBounces(BounceType::Transmission) == 1))
        {
            setIndirectSurfaceGuideData(path.guideData, sd, bsdfProperties);
        }

        if (kOutputNRDData)
        {
            const uint2 pixel = path.getPixel();
            const uint outSampleIdx = params.getSampleOffset(pixel, sampleOffset) + path.getSampleIdx();

            setNRDPrimaryHitEmission(outputNRD, kUseNRDDemodulation, path, pixel, isPrimaryHit, attenuatedEmission);
            setNRDPrimaryHitReflectance(outputNRD, kUseNRDDemodulation, path, pixel, isPrimaryHit, sd, bsdfProperties);
            setNRDPrimaryHitSeparatedRadiance(outputNRD, kUseNRDDemodulation, path, isPrimaryHit, outSampleIdx);

            setNRDSampleHitDist(outputNRD, path, outSampleIdx);
            setNRDSampleEmission(outputNRD, kUseNRDDemodulation, path, outSampleIdx, isPrimaryHit, attenuatedEmission, wasDeltaOnlyPathBeforeScattering);
            setNRDSampleReflectance(outputNRD, kUseNRDDemodulation, path, outSampleIdx, isPrimaryHit, sd, bsdfProperties, lobeTypes, wasDeltaOnlyPathBeforeScattering);
        }

        // Check if this is the last path vertex. Don't terminate at delta vertices as it messes with NRC predictions).
        const bool isLastVertex = hasFinishedSurfaceBounces(path) && (!kUseNRC || !path.isDelta());

        // Handle last vertex (indirect illumination from cache).
        if (isLastVertex) handleLastSurfaceVertex(path, prevThp, sd, bsdfProperties);

        // Terminate if this is the last path vertex and light sampling already completely sampled incident radiance.
        if (kUseNEE && !kUseMIS && isLastVertex && path.isLightSamplable()) valid = false;

        // Terminate caustics paths.
        if (kDisableCaustics && path.getBounces(BounceType::Diffuse) > 0 && path.isSpecular()) valid = false;

        if (!valid)
        {
            path.terminate();
        }
    }

    /** Handle the case when a scatter ray hits a volume.
        After handling the hit, a new scatter ray is generated or the path is terminated.
        \param[in,out] path The path state.
        \param[in,out] vq Visibility query.
    */
    void handleVolumeHit<VisibilityQuery : IVisibilityQuery>(inout PathState path, VisibilityQuery vq)
    {
        // Upon miss in presence of volumes:
        // - Sample light(s) using shadow rays
        // - Sample scatter direction

        const VolumeHit volumeHit = path.hit.getVolumeHit();

        // TODO compute emission through volumes.

        if (path.getBounces(BounceType::Volume) >= kMaxVolumeBounces)
        {
            path.terminate();
            return;
        }

        // Terminate after scatter ray on last vertex has been processed.
        if (hasFinishedSurfaceBounces(path))
        {
            path.terminate();
            return;
        }

        // Russian roulette to terminate paths early.
        if (kUseRussianRoulette)
        {
            if (terminatePathByRussianRoulette(path, sampleNext1D(path.sg))) return;
        }

        // Compute origin for rays traced from this path vertex.
        path.origin += volumeHit.t * path.dir;

        // Sample next-event estimation.
        path.setLightSampled(false, false);
        if (kUseNEE && path.interiorList.isEmpty())
        {
            PathVertex vertex = PathVertex(path.getVertexIndex(), path.origin);

            LightSample ls;
            if (generateLightSample(vertex, true, true, path.sg, ls))
            {
                // Evaluate phase function.
                const HenyeyGreensteinPhaseFunction pf = HenyeyGreensteinPhaseFunction(volumeHit.g);

                // Apply MIS weight.
                if (kUseMIS && ls.lightType != (uint)LightType::Analytic)
                {
                    float phaseFunctionPdf = pf.evalPdf(-path.dir, ls.dir);
                    ls.Li *= evalMIS(1, ls.pdf, 1, phaseFunctionPdf);
                }

                float3 Lr = pf.eval(-path.dir, ls.dir) * ls.Li;
                if (any(Lr > 0.f))
                {
                    const Ray ray = ls.getVisibilityRay();
                    logTraceRay(PixelStatsRayType::Visibility);
                    if (vq.traceVisibilityRay(ray))
                    {
                        Lr *= gridVolumeSampler.evalTransmittance(ray, path.sg);
                        addToPathContribution(path, Lr);
                    }
                }
            }
            path.setLightSampled(true, true);
        }

        // Sample scatter ray.
        bool valid = false;
        if (path.interiorList.isEmpty())
        {
            // Heterogeneous volumes have `g` on the volumeHit
            valid = generateVolumeScatterRay(path, HenyeyGreensteinPhaseFunction(volumeHit.g));
        }
        else
        {
            // Homogeneous volumes get properties from the interiorList
            let material = gScene.materials.getMaterial(path.interiorList.getTopMaterialID());
            VolumeProperties vp = material.getHomogeneousVolumeProperties(gScene.materials, path.interiorList.getTopMaterialID());

            if (kSupportPathStateVolumeProperties && path.hasVolumeProperties())
            {
                vp.sigmaA = path.sigmaA;
                vp.sigmaS = path.sigmaS;
            }

            valid = generateVolumeScatterRay(path, vp.phaseFunction);
        }

        if (!valid)
        {
            path.terminate();
        }
    }

    /** Handle hit on specular materials.
        After handling the hit, a new scatter ray is generated or the path is terminated.
        \param[in,out] path The path state.
    */
    void handleSpecularHit(inout PathState path)
    {
        // Upon hit:
        // - Load vertex/material data
        // - Compute volume absorption
        // - Add emitted radiance
        // - Sample scatter ray or terminate

        const bool isPrimaryHit = path.getVertexIndex() == 1;
        float lodValue = path.getVertexIndex() > 1 ? params.lodBias : 0.f;
        let lod = ExplicitLodTextureSampler(lodValue);

        // Load shading data. This is a long latency operation.
        ShadingData sd = loadShadingData(path.hit, path.origin, path.dir, lod);

        // Reject false hits in nested dielectrics.
        if (!handleNestedDielectrics(sd, path)) return;

        logPathVertex();

        // Create material instance and query its properties.
        let hints = getMaterialInstanceHints(path.hit, isPrimaryHit);
        let mi = gScene.materials.getMaterialInstance(sd, lod, hints);
        BSDFProperties bsdfProperties = mi.getProperties(sd);
        warpProfile(WarpProfilerEvent::MaterialCreate, path.getVertexIndex(), sd.materialID);

        if (kUseScreenSpaceReSTIR && ScreenSpaceReSTIR::kUseGI && path.getVertexIndex() == 1)
        {
            path.reSTIRGIData.isDisabled = true;
        }

        // Disable specular lobes if caustics are disabled and path already contains non-specular vertex.
        bool isSpecular = bsdfProperties.roughness <= params.specularRoughnessThreshold;
        if (kDisableCaustics && path.getBounces(BounceType::Diffuse) > 0 && isSpecular)
        {
            sd.mtl.setActiveLobes((uint)LobeType::Diffuse);
        }

        // Optionally disable emission inside volumes.
        if (!kUseLightsInDielectricVolumes && path.isInsideDielectricVolume())
        {
            bsdfProperties.emission = float3(0.f);
        }

        // Add emitted radiance.
        addToPathContribution(path, bsdfProperties.emission);

        float3 attenuatedEmission = path.thp * bsdfProperties.emission;
        const bool wasDeltaOnlyPathBeforeScattering = path.isDeltaOnlyPath(); // Always true

        // Terminate after scatter ray on last vertex has been processed.
        if (hasFinishedSurfaceBounces(path))
        {
            path.terminate();
            return;
        }

        // Compute origin for rays traced from this path vertex.
        path.origin = sd.computeRayOrigin();

        // Generate the next path segment or terminate.
        bool valid = generateScatterRay(sd, mi, path);

        if (kOutputNRDData)
        {
            const uint2 pixel = path.getPixel();
            const uint outSampleIdx = params.getSampleOffset(pixel, sampleOffset) + path.getSampleIdx();

            setNRDPrimaryHitEmission(outputNRD, kUseNRDDemodulation, path, pixel, isPrimaryHit, attenuatedEmission);

            setNRDSampleHitDist(outputNRD, path, outSampleIdx);
            setNRDSampleEmission(outputNRD, kUseNRDDemodulation, path, outSampleIdx, isPrimaryHit, attenuatedEmission, wasDeltaOnlyPathBeforeScattering);

            // Simply clear sample reflectance on primary hit (it will be overridden by non-delta hit function, if necessary).
            if (isPrimaryHit)
            {
                outputNRD.sampleReflectance[outSampleIdx] = 1.f;
            }
        }

        // Terminate caustics paths.
        if (kDisableCaustics && path.getBounces(BounceType::Diffuse) > 0 && path.isSpecular()) valid = false;

        if (!valid)
        {
            path.terminate();
        }
    }

    /** Handle the case when a scatter ray misses the scene.
        \param[in,out] path The path state.
    */
    void handleMiss(inout PathState path)
    {
        // Upon miss:
        // - Compute MIS weight if previous path vertex sampled a light
        // - Evaluate environment map
        // - Write guiding data
        // - Terminate the path

        // Check if the scatter event is samplable by the light sampling technique.
        const bool isLightSamplable = path.isLightSamplable();

        // Add env radiance.
        bool computeEnv = kUseEnvLight && (!kUseNEE || kUseMIS || !path.isLightSampled() || !isLightSamplable);

        // Disable all constributions to direct illumination if requested.
        if (kDisableDirectIllumination && path.getVertexIndex() == 2) computeEnv = false;

        // With screen space ReSTIR enabled, we sample the full direct illumination
        // contribution on the primary hit. Skip any additional contribution on the
        // secondary hit unless it comes from a scatter event that ReSTIR cannot handle,
        // such as transmission, delta or volume scattering events.
        if (kUseScreenSpaceReSTIR && ScreenSpaceReSTIR::kUseDI && path.getVertexIndex() == 2 && !path.isTransmission() && !path.isDelta() && !path.isVolume()) computeEnv = false;

        float3 emitterRadiance = 0.f;

        if (computeEnv)
        {
            logPathVertex();

            float misWeight = 1.f;
            if (kUseNEE && kUseMIS && path.isLightSampled() && isLightSamplable)
            {
                // If NEE and MIS are enabled, and we've already sampled the env map,
                // then we need to evaluate the MIS weight here to account for the remaining contribution.

                // Evaluate PDF, had it been generated with light sampling.
                float lightPdf = getEnvMapSelectionProbability() * envMapSampler.evalPdf(path.dir);

                // Compute MIS weight by combining this with BSDF sampling.
                // Note we can assume path.pdf > 0.f since we shouldn't have got here otherwise.
                misWeight = evalMIS(1, path.pdf, 1, lightPdf);
            }

            float3 Le = envMapSampler.eval(path.dir);
            emitterRadiance = misWeight * Le;
            addToPathContribution(path, emitterRadiance);

            if (kOutputGuideData && path.getVertexIndex() == 2
                && (path.getBounces(BounceType::Specular) == 1
                || path.getBounces(BounceType::Transmission) == 1))
            {
                // Compress dynamic range similar to UE4.
                float3 compressedColor = pow(Le / (Le + 1.0f), 0.454545f);
                path.guideData.setIndirectAlbedo(compressedColor);
                path.guideData.setReflectionPos(path.dir * kEnvMapDepth);
            }
        }

        if (kOutputGuideData && path.getVertexIndex() == 1)
        {
            path.guideData.setGuideNormal(-path.dir);
        }

        if (kOutputNRDData)
        {
            const uint outSampleIdx = params.getSampleOffset(path.getPixel(), sampleOffset) + path.getSampleIdx();
            setNRDSampleHitDist(outputNRD, path, outSampleIdx);
        }

        if (kUseNRC) path.nrcData.setExitedScene(true);

#if defined(DELTA_REFLECTION_PASS)
        if (path.isDeltaReflectionPrimaryHit())
        {
            writeNRDDeltaReflectionGuideBuffers(outputNRD, kUseNRDDemodulation, path.getPixel(), 0.f, path.thp * emitterRadiance, -path.dir, 0.f, kNRDInvalidPathLength, kNRDInvalidPathLength, 0.f);
        }
        else
        {
            writeNRDDeltaReflectionGuideBuffers(outputNRD, kUseNRDDemodulation, path.getPixel(), 0.f, 0.f, -path.dir, 0.f, kNRDInvalidPathLength, kNRDInvalidPathLength, 0.f);
        }
#elif defined(DELTA_TRANSMISSION_PASS)
        if (path.isDeltaTransmissionPath())
        {
            writeNRDDeltaTransmissionGuideBuffers(outputNRD, kUseNRDDemodulation, path.getPixel(), 0.f, path.thp * emitterRadiance, -path.dir, 0.f, kNRDInvalidPathLength, 0.f, path.origin + path.dir * kNRDInvalidPathLength);
        }
        else
        {
            writeNRDDeltaTransmissionGuideBuffers(outputNRD, kUseNRDDemodulation, path.getPixel(), 0.f, 0.f, -path.dir, 0.f, kNRDInvalidPathLength, 0.f, 0.f);
        }
#endif


        if (kUseScreenSpaceReSTIR && ScreenSpaceReSTIR::kUseGI && path.getVertexIndex() == 2)
        {
            path.reSTIRGIData.position = path.origin + path.dir * 1e10;
            path.reSTIRGIData.normal = -path.dir;
        }

        path.terminate();
    }

    /** Write path contribution to output buffer.
    */
    void writeOutput(inout PathState path)
    {
        assert(!any(isnan(path.L)));

        // Log path length.
        logPathLength(getTerminatedPathLength(path));

        const uint2 pixel = path.getPixel();
        const uint outIdx = params.getSampleOffset(pixel, sampleOffset) + path.getSampleIdx();

        if (kUseNRC)
        {
            const uint2 frame = params.frameDim;
            const uint sampleIndex = path.getSampleIdx();
            const uint pathIndex = NRC::PathInfo::getIndex(frame, pixel, sampleIndex);
            // Only create cache query for Q-learning if the last vertex throughput is non-zero
            if (path.nrcData.isForTraining() && path.nrcData.vertexCount > 0)
            {
                const uint vertexIndex = path.nrcData.vertexCount - 1;
                const uint arrayIndex = NRC::PathVertex::getIndex(nrc.kTrainingResolution, pixel, sampleIndex, vertexIndex, nrc.kTrainingStride.xy, nrc.kMaxPathBounces);
                nrc.pathVertices[arrayIndex].update(path.nrcData.trainingVertexRadiance, path.nrcData.trainingVertexThroughput);

                // Create self-training records for all training paths, including unbiased ones.
                // Without self-training, each training vertex position within the path would matter.
                // Vertices closer to the tail end would receive less indirect illumination, since there
                // are less following vertices, than those closer to the head.
                // An alternative would be to condition the network prediction on the vertex index, but
                // this complicates the task of the network.
                if (!path.nrcData.hasExitedScene())
                {
                    NRC::PathVertex vertex = NRC::PathVertex::createFromPacked(nrc.pathVertices[arrayIndex], path.nrcData.trainingVertexRadiance, path.nrcData.trainingVertexThroughput);
                    path.nrcData.queryBufferIndexTraining = nrc.counters.increment((uint)NRCCounters::kQueries);
                    nrc.query.radianceParams[path.nrcData.queryBufferIndexTraining] = NRC::RadianceParams::createFrom(vertex);
                }
            }
            nrc.pathInfo[pathIndex] = NRC::PathInfo::createFromPathState(path);
        }

        if (kSamplesPerPixel == 1)
        {
            // Write color directly to frame buffer.
            float3 color;
            if (kUseScreenSpaceReSTIR && ScreenSpaceReSTIR::kUseGI && !path.reSTIRGIData.isDisabled) color = path.reSTIRGIData.L0;
            else color = path.L;
            outputColor[pixel] = float4(color, 1.f);
        }
        else
        {
            // Write color to per-sample buffer.
            sampleColor[outIdx].set(path.L);
        }

        if (kOutputGuideData)
        {
            sampleGuideData[outIdx] = path.guideData;
        }

        if (kOutputNRDData)
        {
            // TODO: Optimize this for 1 SPP. It doesn't have to go through resolve pass like the color above.
            NRDRadiance data = {};

            if (path.isDiffusePrimaryHit()) data.setPathType(NRDPathType::Diffuse);
            else if (path.isSpecularPrimaryHit()) data.setPathType(NRDPathType::Specular);
            else if (path.isDeltaReflectionPrimaryHit()) data.setPathType(NRDPathType::DeltaReflection);
            else if (path.isDeltaTransmissionPath()) data.setPathType(NRDPathType::DeltaTransmission);
            else data.setPathType(NRDPathType::Residual);

            data.setRadiance(path.L);

            outputNRD.sampleRadiance[outIdx] = data;
        }

        if (kUseScreenSpaceReSTIR && ScreenSpaceReSTIR::kUseGI)
        {
            float3 sampleRadiance = path.reSTIRGIData.L0.x >= 0 ? path.L - path.reSTIRGIData.L0 : 0;
            screenSpaceReSTIR.setGIInitialSample(
                pixel,
                path.reSTIRGIData.creationPoint,
                path.reSTIRGIData.creationNormal,
                path.reSTIRGIData.position,
                path.reSTIRGIData.normal,
                sampleRadiance);
        }
    }
};
