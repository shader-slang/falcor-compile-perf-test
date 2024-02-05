/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
import Scene.RaytracingInline;
import Utils.Timing.GpuTimer;
import ScreenSpaceReSTIR.ScreenSpaceReSTIR;
import InternalPathTracer;
import InternalPathTracerNRD;
import WorkQueues;

ParameterBlock<InternalPathTracer> gPathTracer;
ParameterBlock<QueueInlineScheduler> gScheduler;

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

/** Persistent threads based scheduler.

    This scheduler runs the path tracer from a set of persistent threads.

    Internally, it fetches paths to trace from input queue(s).
    Depending on the configuration, after each iteration, if a path is terminated,
    new path(s) will be loaded from the queue(s) to keep full occupancy.

    The queuing operations are reduced horizontally using wave intrinsics to
    reduce the number of atomics.
*/
struct QueueInlineScheduler
{
    WorkQueues queues; ///< Path queues and atomic counters.

#if defined(SPECULAR_PASS)
    static const uint kPathRegenerationCount = PATH_REGENERATION_COUNT_SPECULAR;
#elif defined(DELTA_REFLECTION_PASS)
    static const uint kPathRegenerationCount = PATH_REGENERATION_COUNT_DELTA_REFLECTION;
#elif defined(DELTA_TRANSMISSION_PASS)
    static const uint kPathRegenerationCount = PATH_REGENERATION_COUNT_DELTA_TRANSMISSION;
#else
    static const uint kPathRegenerationCount = PATH_REGENERATION_COUNT_MAIN;
#endif

    /** Trace active paths until at least one thread in warp has terminated.
        \param[in,out] path The path state.
        \return True if the path was enqueued for later processing, in which case its result isn't ready to be written out.
    */
    bool tracePath(inout PathState path)
    {
        // Iterate over path vertices until we break out of the loop to fetch more paths.
        bool enqueued = false;
        bool continuePath = false;
        while (true)
        {
            // Handle surface hit.
            if (path.isActive() && path.isHit())
            {
                VisibilityQuery vq;
#if defined(SPECULAR_PASS)
                gPathTracer.handleSpecularHit(path);
#elif defined(DELTA_REFLECTION_PASS)
                gPathTracer.handleDeltaReflectionHit(path);
#elif defined(DELTA_TRANSMISSION_PASS)
                gPathTracer.handleDeltaTransmissionHit(path);
#else
                gPathTracer.handleHit(path, vq);
#endif
            }

            // Move to the next hit.
            ClosestHitQuery chq;
            if (path.isActive() && path.isHit()) gPathTracer.nextHit(path, chq);

#if defined(SPECULAR_PASS)
            // Inspect hit and push non-specular to general queue.
            if (path.isActive() && path.isHit())
            {
                if (!InternalPathTracer::useSpecularQueue(path.hit))
                {
                    uint writeIdx = queues.incrementCounter((uint)Counters::kGeneralQueued);
                    queues.generalQueue[writeIdx] = path.encodeSpecularPath();
                    path.terminate();
                    enqueued = true;
                }
            }
#endif

            // Check if this path needs to be continued.
            continuePath = path.isActive() && path.isHit();

            // Scheduling logic to decide when to break out of the while loop to fetch more work.
            if (kPathRegenerationCount <= 1)
            {
                if (WaveActiveAnyTrue(!continuePath)) break;
            }
            else if (kPathRegenerationCount >= 32)
            {
                if (WaveActiveAllTrue(!continuePath)) break;
            }
            else
            {
                if (WaveGetLaneCount() - WaveActiveCountBits(continuePath) >= kPathRegenerationCount) break;
            }
        }

        // At this point, at least one thread has a path that is terminated or a miss.
        // Handle the miss and terminate path.
        if (!continuePath && path.isActive())
        {
            gPathTracer.handleMiss(path);
        }

        // At this point, the path is either terminated or a valid hit.
        // Return the caller to start new paths in the empty threads.
        // The caller should only write out result if the path wasn't enqueued for later processing.
        return enqueued;
    }

    /** Load new paths from the input queue for all available threads.
        \return True if queue is empty after this operation, false otherwise. The return value is uniform across the warp.
    */
    bool loadFromInputQueue(const uint queueSize, inout PathState path)
    {
        bool queueEmpty = false;
        if (path.isTerminated())
        {
            uint newValue;
#if defined(SPECULAR_PASS)
            uint idx = queues.incrementCounter((uint)Counters::kSpecularPathsProcessed, newValue);
#elif defined(DELTA_REFLECTION_PASS)
            uint idx = queues.incrementCounter((uint)Counters::kDeltaReflectionPathsProcessed, newValue);
#elif defined(DELTA_TRANSMISSION_PASS)
            uint idx = queues.incrementCounter((uint)Counters::kDeltaTransmissionPathsProcessed, newValue);
#else
            uint idx = queues.incrementCounter((uint)Counters::kGeneralPathsProcessed, newValue);
#endif
            if (idx < queueSize)
            {
#if defined(SPECULAR_PASS)
                uint pathID = queues.specularPaths.Load(idx * 4);
#elif defined(DELTA_REFLECTION_PASS)
                uint pathID = queues.deltaReflectionPaths.Load(idx * 4);
#elif defined(DELTA_TRANSMISSION_PASS)
                uint pathID = queues.deltaTransmissionPaths.Load(idx * 4);
#else
                uint pathID = queues.generalPaths.Load(idx * 4);
#endif
                gPathTracer.generatePath(pathID, path);
                gPathTracer.setupPathLogging(path);
            }
            queueEmpty = newValue >= queueSize;
        }
        return WaveActiveAnyTrue(queueEmpty);
    }

    /** Load new paths from the general queue for all available threads.
        \return True if queue is empty after this operation, false otherwise. The return value is uniform across the warp.
    */
    bool loadFromGeneralQueue(const uint queueSize, inout PathState path)
    {
        bool queueEmpty = false;
        if (path.isTerminated())
        {
            uint newValue;
            uint idx = queues.incrementCounter((uint)Counters::kGeneralQueuedProcessed, newValue);
            if (idx < queueSize)
            {
                gPathTracer.generateSpecularPath(queues.generalQueue[idx], path);
                gPathTracer.setupPathLogging(path);
            }
            queueEmpty = newValue >= queueSize;
        }
        return WaveActiveAnyTrue(queueEmpty);
    }

    /** Entry point for persistent threads.
    */
    void run()
    {
        // Fetch the total path counts.
#if defined(SPECULAR_PASS)
        uint totalPathCount = queues.counters.Load((uint)Counters::kSpecularPaths * 4);
#elif defined(DELTA_REFLECTION_PASS)
        uint totalPathCount = queues.counters.Load((uint)Counters::kDeltaReflectionPaths * 4);
#elif defined(DELTA_TRANSMISSION_PASS)
        uint totalPathCount = queues.counters.Load((uint)Counters::kDeltaTransmissionPaths * 4);
#else
        uint totalPathCount = queues.counters.Load((uint)Counters::kGeneralPaths * 4);
        uint totalQueuedPathCount = queues.counters.Load((uint)Counters::kGeneralQueued * 4);
#endif

        // Create live state. This will be populated as we load paths.
        PathState path = {};
        GpuTimer timer;
        bool firstIteration = true;

        while (true)
        {
            if (InternalPathTracer::kOutputTime && path.isTerminated()) timer.start();

            // Load new paths for all available threads.
            if (totalPathCount)
            {
                if (loadFromInputQueue(totalPathCount, path)) totalPathCount = 0;
            }
#if !defined(SPECULAR_PASS) && !defined(DELTA_REFLECTION_PASS) && !defined(DELTA_TRANSMISSION_PASS)
            else if (totalQueuedPathCount)
            {
                if (loadFromGeneralQueue(totalQueuedPathCount, path)) totalQueuedPathCount = 0;
            }
#endif
            else
            {
                // Terminate warp when queues are empty and all paths have terminated.
                if (WaveActiveAllTrue(path.isTerminated())) return;
            }

            // Record number of threads that were used to process work.
            // This is mostly to see that we're launching enough persistent threads to fill the machine.
            if (InternalPathTracer::kEnableStats)
            {
                if (firstIteration && WaveActiveAnyTrue(!path.isTerminated()))
                {
#if defined(SPECULAR_PASS)
                    queues.incrementCounter((uint)Counters::kThreadCountSpecular);
#elif defined(DELTA_REFLECTION_PASS)
                    queues.incrementCounter((uint)Counters::kThreadCountDeltaReflection);
#elif defined(DELTA_TRANSMISSION_PASS)
                    queues.incrementCounter((uint)Counters::kThreadCountDeltaTransmission);
#else
                    queues.incrementCounter((uint)Counters::kThreadCountGeneral);
#endif
                    firstIteration = false;
                }
            }

            // At this point, we have normally full thread occupancy.
            // This is not guaranteed though due to tail effects at the end of the queues.
            // It is even possible that no paths are active if we were at the end of a queue.
            if (!path.isTerminated())
            {
                // Follow the active paths until at least one path in the warp has terminated.
                bool enqueued = tracePath(path);

                // Write result for terminated paths that were not enqueued to be continued later.
                if (path.isTerminated() && !enqueued)
                {
#if !defined(DELTA_REFLECTION_PASS) && !defined(DELTA_TRANSMISSION_PASS)
                    gPathTracer.writeOutput(path);
#endif
                }

                if (InternalPathTracer::kOutputTime && path.isTerminated())
                {
                    InterlockedAdd(gPathTracer.outputTime[path.getPixel()], timer.getElapsed());
                }
            }
        }
    }
}

/** Compute shader launching the inline path tracer.
    The dispatch dimension is over persistent thread id (X).
*/
[numthreads(256, 1, 1)]
void main()
{
    gScheduler.run();
}
