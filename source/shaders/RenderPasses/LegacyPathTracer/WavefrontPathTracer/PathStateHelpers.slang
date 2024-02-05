/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
import Utils.Math.PackedFormats;
__exported import Rendering.Materials.InteriorList;
__exported import RenderPasses.LegacyPathTracer.PathTracer.StaticParams;
__exported import RenderPasses.LegacyPathTracer.PathTracer.PathTracerParams;
__exported import RenderPasses.LegacyPathTracer.WavefrontPathTracer.PathState;

/** Load the current path state from memory.
    Only the data that is relevant for the current path is loaded to save bandwidth.
*/
PathData loadPathState(
    const uint idx,
#ifdef FINAL_ITERATION
    StructuredBuffer<uint> pathFlags,
    StructuredBuffer<PathState> pathState,
    StructuredBuffer<PathRays> pathRays,
    StructuredBuffer<InteriorList> pathInteriorList,
#else
    RWStructuredBuffer<uint> pathFlags,
    RWStructuredBuffer<PathState> pathState,
    RWStructuredBuffer<PathRays> pathRays,
    RWStructuredBuffer<InteriorList> pathInteriorList,
#endif
    StructuredBuffer<PackedHitInfo> pathHit
)
{
    PathData path = {};

    // Load the path flags to see what other data we need to load.
    uint flagsAndLength = pathFlags[idx];
    path.flags = flagsAndLength & ((1u << kMaxPathFlagsBits) - 1);
    path.length = (flagsAndLength >> kMaxPathFlagsBits) & ((1u << kMaxPathLengthBits) - 1);
    path.nonSpecularBounces = (flagsAndLength >> (kMaxPathFlagsBits + kMaxPathLengthBits)) & ((1u << kMaxPathLengthBits) - 1);

    if (path.isTerminated())
        return path;

    // Scatter ray.
    if (path.hasScatterRay())
    {
        path.dir = pathRays[idx].getScatterDir();
        path.thp = pathState[idx].thp;
        path.origin = pathRays[idx].origin;

        // The pdf and normal are stored only if MIS is enabled, because that's the only thing we need them for currently.
        if (kUseMIS)
        {
            path.pdf = pathState[idx].pdf;
            path.normal = decodeNormal2x16(pathState[idx].packedNormal);
        }

        if (path.flags & uint(PathFlags::scatterHit))
        {
            path.hit = HitInfo(pathHit[idx]);
        }

        if (kUseNestedDielectrics)
        {
            path.interiorList = pathInteriorList[idx];
        }

#ifndef FINAL_ITERATION
        // Load sample generator state only if path continues.
        // We do not need random numbers for computing the contribution on the final iteration.
        path.sg = pathState[idx].sg;
#endif
    }

    // Shadow rays.
    // Load the contribution only if ray is active and the light was visible.
    // We directly accumulate the contribution here to reduce the register pressure.
    // TODO: It would be a bit cleaner to accumulate the shadow ray contributions already in the trace pass.
    // However, it was slightly slower than to do it here. In cases where MIS is disabled, it'd be faster though as it
    // allows us to skip the last process paths pass altogether (but MIS off is maybe not an interesting case to optimize for).
    for (uint i = 0; i < kLightSamplesPerVertex; i++)
    {
        if (path.hasShadowRay(i))
        {
            path.L += pathState[idx].Lr[i];
        }
    }

    return path;
}

/** Save the new path state to memory.
    Only the data that is relevant for the current path is saved to save bandwidth.
    We can assume this function is not called on paths that were already terminated when loaded.
*/
void savePathState(
    const PathData path,
    const uint idx,
    RWStructuredBuffer<uint> pathFlags,
    RWStructuredBuffer<PathState> pathState,
    RWStructuredBuffer<PathRays> pathRays,
    RWStructuredBuffer<InteriorList> pathInteriorList,
    RWStructuredBuffer<float3> pathTransmission
)
{
    // Save the updated path flags.
    pathFlags[idx] =
        (path.nonSpecularBounces << (kMaxPathFlagsBits + kMaxPathLengthBits)) | (path.length << kMaxPathFlagsBits) | path.flags;

    if (path.isTerminated())
        return;

    pathRays[idx].origin = path.origin;

    // Scatter ray.
    if (path.hasScatterRay())
    {
        pathRays[idx].setScatterDir(path.dir);
        pathState[idx].thp = path.thp;

        // Note that the pdf can be costly, so we store it only if we'll need it (currently only when MIS is enabled),
        // and rely on dead code elimination to remove the computations otherwise.
        // The same goes for the normal. It's already computed, but it's unnecessary to take the bandwidth cost if we won't need it.
        if (kUseMIS)
        {
            pathState[idx].pdf = path.pdf;
            pathState[idx].packedNormal = encodeNormal2x16(path.normal);
        }

        // Only save the sample generator state if we'll need it, i.e., if we are following the path and are not at the last path vertex.
        if (path.length < kMaxBounces)
        {
            pathState[idx].sg = path.sg;
        }

        if (path.isTransmission())
        {
            pathTransmission[idx] = path.transmissionOrigin;
        }

        if (kUseNestedDielectrics)
        {
            pathInteriorList[idx] = path.interiorList;
        }
    }
}
