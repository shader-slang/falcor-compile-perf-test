/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"

import Scene.Scene;
import Scene.Camera.Camera;
import Scene.Volume.Grid;
import Scene.Volume.GridVolume;
import Utils.Math.MathHelpers;
import Utils.Color.ColorHelpers;
import Utils.Geometry.IntersectionHelpers;
import Utils.Sampling.SampleGenerator;
import Rendering.Lights.EnvMapSampler;
import Rendering.Volumes.GridVolumeSampler;
import Rendering.Volumes.PhaseFunction;

import Params;
import PathData;
import WorkQueue;

// Uniform data.
cbuffer PerFrameCB
{
    uint gSeed;
    WorkQueue queue;
    GridVolumeSampler gGridVolumeSampler;
    EnvMapSampler gEnvMapSampler;
}

// Per-Sample buffers.
RWStructuredBuffer<ColorData> gSampleColor;
Texture2D<uint> gSampleOffsets;

float powerHeuristic(const float a, const float b)
{
    return (a * a) / (a * a + b * b);
}

/** Initialize PathState struct for new camera ray encoded in pathID.
    \param[in] pathID Encoded pixel and current spp.
    \param[in] near Distance to camera where to start traversal from.
    \param[out] path Populated PathState struct ready for traversal.
*/
void generatePath(const uint pathID, const float near, out PathState path)
{
    const uint2 pixel = PathState::decodePixel(pathID);
    const uint sample = PathState::decodeSample(pathID);
    const Ray ray = gScene.camera.computeRayPinhole(pixel, kResolution); // TODO: unqiue AA/DOF samples for multiple spp?

    path = {};
    path.flags = uint(PathFlags::active);
    path.id = pathID;
    path.wpos = ray.origin + near * ray.dir;
    path.wdir = ray.dir;
    path.throughput = float3(1.f);
    path.pdf = 1.f;
    path.sg = SampleGenerator(pixel, gSeed * kSamplesPerPixel + sample);

    if (kComputeStats)
        queue.incrementCounter((uint)Counters::kPathsGenerated);
}

/** Load new paths from the input queue for all available threads.
    \return True if queue is empty after this operation, false otherwise. The return value is uniform across the warp.
*/
bool loadFromVolumeQueue(const uint queueSize, inout PathState path)
{
    bool queueEmpty = false;
    if (path.isTerminated())
    {
        uint newValue;
        uint idx = queue.incrementCounter((uint)Counters::kVolumePathsProcessed, newValue);
        if (idx < queueSize)
        {
            uint pathID = queue.volumePathIDs.Load(idx * 4);
            const float near = queue.volumePathDistances.Load(idx * 4);
            generatePath(pathID, near, path);
        }
        queueEmpty = newValue >= queueSize;
    }
    return WaveActiveAnyTrue(queueEmpty);
}

/** Test for collisions against the volume.
    \param[in,out] path Current state of the path (updated accordingly).
    \return True if a collision with the volume was sampled, false otherwise.
*/
bool sampleVolume(inout PathState path)
{
    if (path.isTerminated() || path.isFree())
        return false;
    // TODO: emission
    GridVolumeSampler::DistanceSample ds;
    const Ray ray = Ray(path.wpos, path.wdir);
    const bool hit = gGridVolumeSampler.sampleDistance(ray, path.sg, ds);
    if (hit)
    {
        // Advance ray to collision and update throughput.
        path.wpos += ds.t * path.wdir;
        path.throughput *= ds.thp;
        path.length++;
    }
    else
    {
        // Missed the volume, or did not find a collision.
        path.setFree();
    }
    return hit;
}

/** Shade using next event estimation for each valid path vertex if enabled.
    \param[in,out] path Current state of the path (updated accordingly).
*/
void shadeNextEventEstimation(inout PathState path)
{
    if (!kUseNEE)
        return;
    // Sample environment light probe.
    EnvMapSample envSample;
    if (gEnvMapSampler.sample(sampleNext2D(path.sg), envSample))
    {
        const HenyeyGreensteinPhaseFunction pf = HenyeyGreensteinPhaseFunction(gScene.getGridVolume(0).getAnisotropy());
        const float3 phase = pf.eval(-path.wdir, envSample.dir);
        float misWeight = 1.f;
        if (kUseMIS)
            misWeight = powerHeuristic(envSample.pdf, pf.evalPdf(-path.wdir, envSample.dir));
        const Ray ray = Ray(path.wpos, envSample.dir);
        const float Tr = gGridVolumeSampler.evalTransmittance(ray, path.sg);
        path.radiance += path.throughput * envSample.Le * misWeight * Tr * phase / envSample.pdf;
    }
}

/** Test for path termination criteria and apply russian roulette if enabled.
    \param[in,out] path Current state of the path (updated accordingly).
    \return True if the path was terminated, false otherwise.
*/
bool checkTerminationAndApplyRR(inout PathState path)
{
    // Early-out check.
    bool terminate = path.length > kMaxBounces;
    // Russian roulette.
    if (kUseRR)
    {
        const float rrVal = luminance(path.throughput);
        if (!terminate && rrVal < .1f)
        {
            const float rrProb = max(0.f, 1.f - rrVal);
            terminate |= sampleNext1D(path.sg) < rrProb;
            path.throughput /= max(1e-8f, 1.f - rrProb);
        }
    }
    if (terminate)
    {
        path.terminate();
        if (kComputeStats)
            queue.incrementCounter((uint)Counters::kRaysTerminated);
    }
    return terminate;
}

/** Scatter each valid path vertex according to the phase function.
    \param[in,out] path Current state of the path (updated accordingly).
*/
void scatterPath(inout PathState path)
{
    const HenyeyGreensteinPhaseFunction pf = HenyeyGreensteinPhaseFunction(gScene.getGridVolume(0).getAnisotropy());
    float3 weight;
    bool valid = pf.sample(-path.wdir, path.wdir, path.pdf, weight, path.sg);
    if (valid)
    {
        path.throughput *= weight;
        if (kComputeStats)
            queue.incrementCounter((uint)Counters::kRaysScattered);
    }
    else
    {
        path.terminate();
        if (kComputeStats)
            queue.incrementCounter((uint)Counters::kRaysTerminated);
    }
}

/** Add contribution from volume misses or paths that have left the volume without earlier termination.
    \param[in,out] path Current state of the path (updated accordingly).
*/
void shadeFreePath(inout PathState path)
{
    if (path.isFree())
    {
        const bool miss = path.length == 0 && kShowBackground;
        const bool shade = path.length > 0 && (!kUseNEE || (kUseNEE && kUseMIS));
        if (miss || shade)
        {
            float weight = 1.f;
            if (path.length > 0 && kUseNEE && kUseMIS)
                weight = powerHeuristic(path.pdf, gEnvMapSampler.evalPdf(path.wdir));
            path.radiance += path.throughput * weight * gEnvMapSampler.eval(path.wdir);
        }
        path.terminate();
    }
}

/** Trace paths through the volume, while optionally breaking out of the loop when occupancy is too low.
    \param[in,out] path Current state of the path (updated accordingly).
    \param[in] regeneratePaths Wheather to enable breaking out of the loop to fetch more work.
*/
void tracePathVolume(inout PathState path, const bool regeneratePaths)
{
    while (sampleVolume(path))
    {
        shadeNextEventEstimation(path);

        if (checkTerminationAndApplyRR(path))
            break;

        scatterPath(path);

        // Scheduling logic to decide when to break out of the while loop to fetch more work.
        const uint emptyLanes = WaveGetLaneCount() - WaveActiveCountBits(!path.isTerminated());
        if (regeneratePaths && emptyLanes >= kPathRegenerationCount)
            break;
    }

    if (kComputeStats && regeneratePaths)
    {
        const uint emptyLanes = WaveGetLaneCount() - WaveActiveCountBits(!path.isTerminated());
        if (WaveGetLaneIndex() < emptyLanes)
            queue.incrementCounter((uint)Counters::kPathsReloaded);
    }

    shadeFreePath(path);
}

// --------------------------------------------------------------
// Main entry point.

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    uint volumePathCount = queue.counters.Load((uint)Counters::kVolumePaths * 4);
    PathState path = {};

    while (true)
    {
        // Load new paths for all available threads.
        if (volumePathCount)
        {
            if (loadFromVolumeQueue(volumePathCount, path))
                volumePathCount = 0;
        }
        else
        {
            // We can immediately terminate here because tracePathVeolume will
            // finish computing all paths when volumePathCount == 0.
            return;
            // Terminate warp when queues are empty and all paths have terminated.
            // if (WaveActiveAllTrue(path.isTerminated())) return;
        }

        // At this point, we have normally full thread occupancy.
        // This is not guaranteed though due to tail effects at the end of the queues.
        // It is even possible that no paths are active if we were at the end of a queue.
        if (!path.isTerminated())
        {
            // Follow the active paths until at least one path in the warp has terminated.
            tracePathVolume(path, volumePathCount > 0);

            // Write result for terminated paths.
            if (path.isTerminated())
            {
                const uint2 pixel = path.getPixel();
                const uint sample = path.getSample();
                const uint outIdx = getSampleOffset(pixel, gSampleOffsets) + sample;
                gSampleColor[outIdx].set(path.radiance);
            }
        }
    }
}
