/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
import Scene.Raytracing; // Only required as we run this as a raytracing shader instead of compute.
import Scene.RaytracingInline;
import Rendering.Utils.PixelStats;
import Utils.Math.PackedFormats;
import Utils.Math.FormatConversion;
import InternalPathTracer;
import InternalPathTracerNRD;
import PathState;
import StaticParams;

import Utils.NVAPI; // TODO(@skallweit) this is currently needed to make g_NvidiaExt visible

ParameterBlock<InternalPathTracer> gPathTracer;
ReorderInlineScheduler gScheduler;

struct DummyPayload { uint dummy; };
struct DummyAttribs { uint dummy; };
struct ClosestHitQuery : IClosestHitQuery
{
    bool traceRay(inout PathState path, const Ray ray, out HitInfo hit, out float hitT)
    {
        SceneRayQuery<kUseAlphaTest> sceneRayQuery;
        return sceneRayQuery.traceRay(ray, hit, hitT, RAY_FLAG_NONE, 0xff);
    }
};

struct VisibilityQuery : IVisibilityQuery
{
    bool traceVisibilityRay(const Ray ray)
    {
        SceneRayQuery<kUseAlphaTest> sceneRayQuery;
        return sceneRayQuery.traceVisibilityRay(ray, RAY_FLAG_NONE, 0xff);
    }
};

/** Reordering scheduler using TraceRayInline + SER/HitObject API.
*/
struct ReorderInlineScheduler
{
    /** Compute the coherence hint for a given path state.
        \param[in] path The path state.
        \return Returns the coherence hints in the 8 least significant bits.
    */
    uint getCoherenceHint(const PathState path)
    {
        // Determine what material type is used.
        // Use materialType = 0 for volume hits, materialType > 0 for other materials.
        uint materialType = 0;
        bool isDeltaSpecular = false;
        bool isEmissive = false;
        if (path.hit.getType() != HitType::Volume)
        {
            const uint materialID = gScene.getMaterialID(path.hit.getInstanceID());
            const MaterialHeader materialHeader = gScene.materials.getMaterialHeader(materialID);
            materialType = 1 + (uint)materialHeader.getMaterialType();
            isDeltaSpecular = materialHeader.isDeltaSpecular();
            isEmissive = materialHeader.isEmissive();
        }

        const PathCoherenceHints pathHints = gPathTracer.getCoherenceHints(path, isDeltaSpecular, isEmissive);

        // MSB = highest priority hint.
        return
            (pathHints.computesEmissive ? 0x1 : 0x0) |
            (pathHints.samplesLight ? 0x2 : 0x0) |
            (pathHints.terminated ? 0x4 : 0x0) |
            (pathHints.terminatedByRussianRoulette ? 0x8 : 0x0) |
            (materialType & 0xf) << 4;
    }

    /** Traces a path and write result to output.
        \param[in] pathID Path ID.
    */
    void tracePath(uint pathID)
    {
        PathState path;
        gPathTracer.generatePath(pathID, path);
        gPathTracer.setupPathLogging(path);

        // Note the primary miss has already been handled by the separate path generation pass
        // the path tracer runs first. Abort execution here to avoid double work.
        if (!path.isHit()) return;

        while (path.isActive())
        {
            if (path.isHit())
            {
                VisibilityQuery vq;
#if defined(DELTA_REFLECTION_PASS)
                gPathTracer.handleDeltaReflectionHit(path);
#elif defined(DELTA_TRANSMISSION_PASS)
                gPathTracer.handleDeltaTransmissionHit(path);
#else
                gPathTracer.handleHit(path, vq);
#endif
            }
            else
            {
                gPathTracer.handleMiss(path);
            }

            if (path.isTerminated()) break;

            {
                ClosestHitQuery chq;
                gPathTracer.nextHit(path, chq);

#if USE_SER
                // Create a HitObject from a HitInfo.
                // The generated HitObject is used for creating the reordering key.
                // Both the hit / miss will use a dummy entry in the shader binding table, as we're only
                // interested in using the geometry data(instance, geometry and primitive index) for sorting.
                // TODO: Move to a utility function when Slang limitation has been resolved. See https://gitlab-master.nvidia.com/nvresearch-gfx/Tools/Falcor/-/issues/1611
                HitObject hitObject;
                RayDesc dummyRay = {};

                if (path.hit.isValid())
                {
                    const GeometryInstanceData instance = gScene.getGeometryInstance(path.hit.getInstanceID());
                    const uint primitiveIndex = path.hit.getPrimitiveIndex();

                    // Setup HitObject.
                    DummyAttribs dummyAttribs = {};
                    hitObject = HitObject::MakeHit(
                        0,                      // HitGroupRecordIndex
                        gScene.rtAccel,         // AccelerationStructure
                        instance.instanceIndex, // InstanceIndex
                        instance.geometryIndex, // GeometryIndex
                        primitiveIndex,         // PrimitiveIndex
                        0,                      // HitKind
                        dummyRay,               // Ray
                        dummyAttribs            // Attributes
                    );
                }
                else
                {
                    hitObject = HitObject::MakeMiss(0, dummyRay);
                }

                // Note: In the past we helped the compiler figuring out live state by
                // packing/unpacking the path state. This is not required with new drivers anymore.
                // PackedPathState packed = path.encode();
                // Reorder threads based on sort key from HitObject and custom coherence hints.
                ReorderThread(hitObject, getCoherenceHint(path), 8);
                // path.decode(packed);
#endif
            }
        }

#if !defined(DELTA_REFLECTION_PASS) && !defined(DELTA_TRANSMISSION_PASS)
        gPathTracer.writeOutput(path);
#endif
    }

    /** Runs the path tracer for a single pixel.
        \param[in] pixel Pixel index.
    */
    void run(uint2 pixel)
    {
        // Determine number of samples to take.
        uint samplesRemaining = kSamplesPerPixel;
        if (kSamplesPerPixel == 0)
        {
            samplesRemaining = gPathTracer.sampleCount[pixel];
        }

        // Loop over samples.
        while (samplesRemaining > 0)
        {
            samplesRemaining -= 1;
            uint pathID = pixel.x | (pixel.y << 12) | (samplesRemaining << 24);
            tracePath(pathID);

#if USE_SER
            // Use SER to compact active threads.
            if (kSamplesPerPixel == 0)
            {
                bool needReorder = WaveActiveAnyTrue(samplesRemaining == 0);
                if (samplesRemaining == 0) break;
                if (needReorder)
                {
                    HitObject hitObject = HitObject::MakeNop();
                    ReorderThread(hitObject);
                }
            }
#endif
        }
    }
}

//
// Shader entry point for ray generation shader.
//

[shader("raygeneration")]
void rayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 frameDim = DispatchRaysDimensions().xy;
    if (all(pixel >= frameDim)) return;

    gScheduler.run(pixel);
}

//
// Shader entry point for dummy miss shader.
//

[shader("miss")]
void dummyMiss(inout DummyPayload payload: SV_RayPayload)
{
}

//
// Shader entry point for dummy closest hit shader.
//

[shader("closesthit")]
void dummyClosestHit(inout DummyPayload payload: SV_RayPayload, DummyAttribs attribs: SV_IntersectionAttributes)
{
}
