/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
import Utils.Attributes;
import Utils.Color.ColorHelpers;
import Utils.Math.MathHelpers;
import Rendering.Lights.EnvMapSampler;
import Rendering.Volumes.GridVolumeSampler;
import RenderPasses.Shared.Denoising.NRDConstants;
import RenderPasses.Shared.Denoising.NRDBuffers;
import ScreenSpaceReSTIR.ScreenSpaceReSTIR;
import NeuralRadianceCache.NeuralRadianceCache;
import LoadShadingData;
import NRDHelpers;
import InternalPathTracer;
import PathState;
import WorkQueues;
import ColorType;
import Params;


// Shared memory variables.
// TODO: Can we declare these inside PathGenerator?
static const uint kWarpCount = (kScreenTileDim.x * kScreenTileDim.y) / 32;

// TODO: Replace explicitly declared size by compile-time constant when it works. For now assume tile is always 16x16!
// See https://gitlab-master.nvidia.com/nvresearch-gfx/Tools/slang/-/issues/87
groupshared uint gSamplesOffset[8 /* kWarpCount */];
groupshared uint gGeneralOffset[8 /* kWarpCount */];
groupshared uint gSpecularOffset[8 /* kWarpCount */];
groupshared uint gDeltaReflectionOffset[8 /* kWarpCount */];
groupshared uint gDeltaTransmissionOffset[8 /* kWarpCount */];

/** Helper struct for generating paths in screen space.

    The primary hits are classified based on material properties and appended
    to different queues. Hits on the background are directly evaluated and written
    to the output buffer, as there is no point in deferring them to a separate queue.

    The dispatch size is one thread group per screen tile. A warp is assumed to be 32 threads.
    Within a thread group, the threads are linearly indexed and mapped to pixels in Morton order.

    Output queues
    -------------

    The output queues have all paths packed tightly, organized by tile.
    The tile order is non-deterministic, but the order within tiles is well-defined.
    The pixels are enumerated in Morton order with all samples for a pixel stored consecutively.

    For each pixel on screen, we determine which queue it should be processed on and append
    one path entry for each sample in that pixel. Note that the number of samples per pixel may vary.

    Output sample buffer
    --------------------

    For each pixel that belongs to the background, and hence does not need to be path traced,
    we directly evaluate the background color and write all samples to the output sample buffer.

    The output sample buffer is organized by tiles in scanline order. Within tiles,
    the pixels are enumerated in Morton order with all samples for a pixel stored consecutively.

    When the number of samples/pixel is not fixed, we additionally write a 2D lookup table,
    for each pixel storing the tile-local offset to where the first sample is stored.
    Based on this information, subsequent passes can easily find the location of a given sample.

*/
struct PathGenerator
{
    // Resources
    PathTracerParams params;                        ///< Runtime parameters.
    GridVolumeSampler gridVolumeSampler;            ///< Grid volume sampler. Only valid when kUseGridVolumes == true.
    WorkQueues queues;                              ///< Work queues we append paths to.
    ScreenSpaceReSTIR screenSpaceReSTIR;            ///< Screen space ReSTIR sampler if kUseScreenSpaceReSTIR == true.
    NRC nrc;                                        ///< Neural radiance cache. Only valid if kUseNRC == true.

    Texture2D<PackedHitInfo> vbuffer;               ///< Fullscreen V-buffer for the primary hits.
    Texture2D<float3> viewDir;                      ///< Optional view direction. Only valid when kUseViewDir == true.
    Texture2D<float> spatioTemporalBlueNoise;       ///< Spatio-temporal blue noise table       
    Texture2D<uint> sampleCount;                    ///< Optional input sample count buffer. Only valid when kSamplesPerPixel == 0.
    RWTexture2D<uint> sampleOffset;                 ///< Output offset into per-sample buffers. Only valid when kSamplesPerPixel == 0.

    RWStructuredBuffer<ColorType> sampleColor;      ///< Output per-sample color if kSamplesPerPixel != 1.
    RWStructuredBuffer<GuideData> sampleGuideData;  ///< Output per-sample guide data.
    NRDBuffers outputNRD;                           ///< Output NRD data.

    RWTexture2D<float4> outputColor;                ///< Output color buffer if kSamplesPerPixel == 1.

    // Render settings that depend on the scene.
    // TODO: Move into scene defines.
    static const bool kUseEnvLight = USE_ENV_LIGHT;
    static const bool kUseGridVolumes = USE_GRID_VOLUMES;

    // Additional specialization.
    static const bool kUseSpecularQueue = USE_SPECULAR_QUEUE;
    static const bool kOutputGuideData = OUTPUT_GUIDE_DATA;

    /** Entry point for path generator.
        \param[in] tileID Tile ID in x and y on screen.
        \param[in] threadIdx Thread index within the tile.
    */
    void execute(const uint2 tileID, const uint threadIdx)
    {
        // Map thread to pixel based on Morton order within tile.
        // The tiles themselves are enumerated in scanline order on screen.
        const uint2 tileOffset = tileID << kScreenTileBits; // Tile offset in pixels.
        const uint2 pixel = deinterleave_8bit(threadIdx) + tileOffset; // Assumes 16x16 tile or smaller. A host-side assert checks this assumption.

        // Process the pixel to determine which queue to use.
        // If we don't hit any surfaces or volumes, then the background will be evaluated and written out directly.
        Ray cameraRay;
        bool useGeneralQueue = false;
        bool useSpecularQueue = false;
        // These queues are used to generate guide buffers (only one path per pixel per queue necessary)
        bool useDeltaReflectionQueue = false;
        bool useDeltaTransmissionQueue = false;
        uint spp = 0;

        // Note: Do not terminate threads for out-of-bounds pixels because we need all threads active for the prefix sum pass below.
        if (all(pixel < params.frameDim))
        {
            // Determine number of samples at the current pixel.
            // This is either a fixed number or loaded from the sample count texture.
            // TODO: We may want to use a nearest sampler to allow the texture to be of arbitrary dimension.
            spp = kSamplesPerPixel > 0 ? kSamplesPerPixel : min(sampleCount[pixel], kMaxSamplesPerPixel);

            // Compute the primary ray.
            cameraRay = gScene.camera.computeRayPinhole(pixel, params.frameDim);
            if (kUseViewDir) cameraRay.dir = -viewDir[pixel];

            // Load the primary hit from the V-buffer.
            const HitInfo hit = HitInfo(vbuffer[pixel]);
            bool hitSurface = hit.isValid();

            // Check for volume hit.
            bool hitVolume = false;
            if (kUseGridVolumes) hitVolume = gridVolumeSampler.intersectsVolumes(cameraRay);

            if (hitSurface || hitVolume)
            {
                // For hits: determine which queue to use.
                // TODO: Generalize this to use queue IDs and WaveMatch() to identify which threads write to which queue, if we start adding more queue types. For now there are only two queues.
                if (kUseSpecularQueue && hitSurface && !hitVolume) useSpecularQueue = InternalPathTracer::useSpecularQueue(hit);
                if (!useSpecularQueue) useGeneralQueue = true;

                // TODO: Material can have textured specularTransmission - is it worth fetching?
                // For now be conservative and spawn paths across the whole screen.
                if (kOutputNRDAdditionalData)
                {
                    useDeltaReflectionQueue = true;
                    useDeltaTransmissionQueue = true;
                }
            }

            // Prepare per-pixel data for screen-space ReSTIR.
            if (kUseScreenSpaceReSTIR)
            {
                bool validSurface = false;
                if (hitSurface)
                {
                    let lod = ExplicitLodTextureSampler(0.f);
                    ShadingData sd = loadShadingData(hit, cameraRay.origin, cameraRay.dir, lod);

                    // Create BSDF instance and query its properties.
                    let hints = getMaterialInstanceHints(hit, true /* primary hit */);
                    let mi = gScene.materials.getMaterialInstance(sd, lod, hints);
                    let bsdfProperties = mi.getProperties(sd);

                    // Check for BSDF lobesTypes that ReSTIR can sample.
                    uint lobesTypes = mi.getLobeTypes(sd);
                    validSurface = (lobesTypes & (uint)LobeType::NonDeltaReflection) != 0;

                    if (validSurface)
                    {
                        // ReSTIR uses a simple material model with only diffuse and specular reflection lobes.
                        // We query the BSDF for the diffuse albedo and specular reflectance, and use their luminances as weights.
                        // TODO: https://gitlab-master.nvidia.com/nvresearch-gfx/Tools/Falcor/-/issues/1406
                        float depth = distance(cameraRay.origin, sd.posW);
                        float diffuseWeight = luminance(bsdfProperties.diffuseReflectionAlbedo);
                        float specularWeight = luminance(bsdfProperties.specularReflectance);
                        screenSpaceReSTIR.setSurfaceData(pixel, sd.computeRayOrigin(), depth, bsdfProperties.guideNormal, sd.faceN, diffuseWeight, specularWeight, bsdfProperties.roughness, lobesTypes);
                    }
                }
                if (!validSurface) screenSpaceReSTIR.setInvalidSurfaceData(pixel);
            }
        }

        // Perform a reduction over the tile to determine the number of paths
        // to append to each queue. This is done via wave ops and shared memory.
        // The write offsets are given by prefix sums over the threads.
        const uint warpIdx = threadIdx >> 5;

        // Calculate the path and sample counts over the warp.
        // The first thread in each warp writes the results to shared memory.
        {
            uint samples = WaveActiveSum(spp);
            uint generalPaths = WaveActiveSum(useGeneralQueue ? spp : 0);
            uint specularPaths = WaveActiveSum(useSpecularQueue ? spp : 0);
            uint deltaReflectionPaths = WaveActiveSum(useDeltaReflectionQueue ? 1 : 0); // Only one path per pixel needed
            uint deltaTransmissionPaths = WaveActiveSum(useDeltaTransmissionQueue ? 1 : 0); // Only one path per pixel needed

            if (WaveIsFirstLane())
            {
                gSamplesOffset[warpIdx] = samples;
                gGeneralOffset[warpIdx] = generalPaths;
                gSpecularOffset[warpIdx] = specularPaths;
                gDeltaReflectionOffset[warpIdx] = deltaReflectionPaths;
                gDeltaTransmissionOffset[warpIdx] = deltaTransmissionPaths;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // Compute the prefix sum over the warp totals in shared memory.
        // The first N threads in the thread group perform this computation.
        if (threadIdx < kWarpCount)
        {
            // Compute the prefix sum over the sample counts.
            uint samples = gSamplesOffset[threadIdx];
            gSamplesOffset[threadIdx] = WavePrefixSum(samples);

            // Allocate space in the path queues.
            // The return values are the global offsets into the respective queues.
            uint generalPaths = gGeneralOffset[threadIdx];
            gGeneralOffset[threadIdx] = queues.addCounter((uint)Counters::kGeneralPaths, generalPaths);

            uint specularPaths = gSpecularOffset[threadIdx];
            gSpecularOffset[threadIdx] = queues.addCounter((uint)Counters::kSpecularPaths, specularPaths);

            uint deltaReflectionPaths = gDeltaReflectionOffset[threadIdx];
            gDeltaReflectionOffset[threadIdx] = queues.addCounter((uint)Counters::kDeltaReflectionPaths, deltaReflectionPaths);

            uint deltaTransmissionPaths = gDeltaTransmissionOffset[threadIdx];
            gDeltaTransmissionOffset[threadIdx] = queues.addCounter((uint)Counters::kDeltaTransmissionPaths, deltaTransmissionPaths);
        }
        GroupMemoryBarrierWithGroupSync();

        if (all(pixel < params.frameDim))
        {
            // Compute the output sample index.
            // For a fixed sample count, the output index is computed directly from the thread index.
            // For a variable sample count, the output index is given by the prefix sum over sample counts.
            const uint outTileOffset = params.getTileOffset(tileID);
            uint outIdx = 0;

            if (kSamplesPerPixel > 0)
            {
                outIdx = outTileOffset + threadIdx * kSamplesPerPixel;
            }
            else
            {
                uint outSampleOffset = gSamplesOffset[warpIdx] + WavePrefixSum(spp);
                outIdx = outTileOffset + outSampleOffset;

                // Write sample offset lookup table. This will be used by later passes.
                sampleOffset[pixel] = outSampleOffset;
            }

            // Append paths to queues.
            uint pathID = (pixel.y << 12) | pixel.x; // TODO: Helper struct for encoding/decoding path ID.

            // Delta reflection and transmission queue can coexist with general or specular queue.
            if (useDeltaReflectionQueue)
            {
                uint dstIdx = gDeltaReflectionOffset[warpIdx] + WavePrefixSum(1);
                queues.deltaReflectionPaths.Store(dstIdx * 4, pathID);
            }

            if (useDeltaTransmissionQueue)
            {
                uint dstIdx = gDeltaTransmissionOffset[warpIdx] + WavePrefixSum(1);
                queues.deltaTransmissionPaths.Store(dstIdx * 4, pathID);
            }

            if (useGeneralQueue)
            {
                uint dstIdx = gGeneralOffset[warpIdx] + WavePrefixSum(spp);
                for (uint i = 0; i < spp; i++)
                {
                    queues.generalPaths.Store((dstIdx + i) * 4, pathID);
                    pathID += (1 << 24); // Increment sampleIdx field.
                }
            }
            else if (useSpecularQueue)
            {
                uint dstIdx = gSpecularOffset[warpIdx] + WavePrefixSum(spp);
                for (uint i = 0; i < spp; i++)
                {
                    queues.specularPaths.Store((dstIdx + i) * 4, pathID);
                    pathID += (1 << 24); // Increment sampleIdx field.
                }
            }
            else
            {
                // Write background pixels.
                writeBackground(pixel, spp, outIdx, cameraRay.dir);
            }
        }
    }

    void writeBackground(const uint2 pixel, const uint spp, const uint outIdx, const float3 dir)
    {
        // Evaluate background color for the current pixel.
        float3 color = float3(0.f);
        if (kUseEnvLight)
        {
            color = gScene.envMap.eval(dir);
        }

        // Write color and denoising guide data for all samples in pixel.
        // For the special case of fixed 1 spp we write the color directly to the output texture.
        if (kSamplesPerPixel == 1)
        {
            outputColor[pixel] = float4(color, 1.f);
        }

        for (uint i = 0; i < spp; i++)
        {
            if (kSamplesPerPixel != 1)
            {
                sampleColor[outIdx + i].set(color);
            }

            if (kOutputGuideData)
            {
                InternalPathTracer::setBackgroundGuideData(sampleGuideData[outIdx + i], dir, color);
            }

            if (kOutputNRDData)
            {
                outputNRD.sampleRadiance[outIdx + i] = {};

                outputNRD.sampleHitDist[outIdx + i] = kNRDInvalidPathLength;
                outputNRD.samplePrimaryHitNEEOnDelta[outIdx + i] = 0.f;
                outputNRD.sampleEmission[outIdx + i] = 0.f;
                outputNRD.sampleReflectance[outIdx + i] = 1.f;
            }
        }

        if (kOutputNRDData)
        {
            outputNRD.primaryHitEmission[pixel] = float4(color, 1.f);
            outputNRD.primaryHitDiffuseReflectance[pixel] = 0.f;
            outputNRD.primaryHitSpecularReflectance[pixel] = 0.f;
        }

        if (kOutputNRDAdditionalData)
        {
            writeNRDDeltaReflectionGuideBuffers(outputNRD, kUseNRDDemodulation, pixel, 0.f, 0.f, -dir, 0.f, kNRDInvalidPathLength, kNRDInvalidPathLength, 0.f);
            writeNRDDeltaTransmissionGuideBuffers(outputNRD, kUseNRDDemodulation, pixel, 0.f, 0.f, -dir, 0.f, kNRDInvalidPathLength, 0.f, 0.f);
        }

        if (kUseNRC)
        {
            uint pathIndex = NRC::PathInfo::getIndex(params.frameDim, pixel, 0);
            for (uint i = 0; i < spp; i++)
            {
                nrc.pathInfo[pathIndex + i].setAsBackground(color);
            }
        }
    }
};

cbuffer CB
{
    PathGenerator gPathGenerator;
}

// TODO: Replace by compile-time uint2 constant when it works in Slang.
// See related issue https://gitlab-master.nvidia.com/nvresearch-gfx/Tools/slang/-/issues/87
[numthreads(256 /* kScreenTileDim.x * kScreenTileDim.y */, 1, 1)]
void main(
    uint3 groupID : SV_GroupID,
    uint3 groupThreadID : SV_GroupThreadID)
{
    gPathGenerator.execute(groupID.xy, groupThreadID.x);
}
