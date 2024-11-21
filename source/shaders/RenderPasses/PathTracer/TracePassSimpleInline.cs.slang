/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
import Scene.RaytracingInline;
import PathTracer;
import PathTracerNRD;

ParameterBlock<PathTracer> gPathTracer;
SimpleInlineScheduler gScheduler;

// struct ClosestHitQuery : IClosestHitQuery
// {
//     bool traceRay(inout PathState path, const Ray ray, out HitInfo hit, out float hitT)
//     {
//         SceneRayQuery<kUseAlphaTest> sceneRayQuery;
//         return sceneRayQuery.traceRay(ray, hit, hitT, RAY_FLAG_NONE, 0xff);
//     }
// };

struct VisibilityQuery : IVisibilityQuery
{
    bool traceVisibilityRay(const Ray ray)
    {
        SceneRayQuery<kUseAlphaTest> sceneRayQuery;
        return sceneRayQuery.traceVisibilityRay(ray, RAY_FLAG_NONE, 0xff);
    }
};

/** Simple scheduler (fullscreen compute pass) using TraceRayInline.
*/
struct SimpleInlineScheduler
{
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
                if (path.isTerminated()) break;

                // ClosestHitQuery chq;
                // gPathTracer.nextHit(path, chq);
            }
            else
            {
                gPathTracer.handleMiss(path);
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
        if (kSamplesPerPixel == 1)
        {
            // Handle fixed 1 spp case.
            uint pathID = pixel.x | (pixel.y << 12);
            tracePath(pathID);
        }
        else if (kSamplesPerPixel > 1)
        {
            // Handle fixed multiple spp case.
            for (uint sampleIdx = 0; sampleIdx < kSamplesPerPixel; ++sampleIdx)
            {
                uint pathID = pixel.x | (pixel.y << 12) | (sampleIdx << 24);
                tracePath(pathID);
            }
        }
        else
        {
            // Handle dynamic spp case.
            uint sampleCount = gPathTracer.sampleCount[pixel];
            for (uint sampleIdx = 0; sampleIdx < sampleCount; ++sampleIdx)
            {
                uint pathID = pixel.x | (pixel.y << 12) | (sampleIdx << 24);
                tracePath(pathID);
            }
        }
    }
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    if (all(pixel >= gPathTracer.params.frameDim)) return;

    gScheduler.run(pixel);
}
