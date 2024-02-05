/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/

/** Declarations for the path state buffer shared between all passes.
    The path state is addressed in blocks of 32 elements stored consecutively in memory.
    The host sets the BLOCK_WIDTH and BLOCK_HEIGHT defines, which are the frame dimensions in blocks.
*/
import Utils.Math.MathHelpers;
__exported import RenderPasses.LegacyPathTracer.PathTracer.PathData;

static const uint3 kBlockDim = { BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_DEPTH };
static const uint kInvalidIndex = 0xffffffff;

/** The in-memory representation of the path state.

    The struct stores some of the inputs/outputs for the wavefront compute passes.
    Other data is in separate buffers to allow more optimal access from the trace passes.
    Note that the state is stored in a structured buffer, so it uses tight packing rules.
    This is typically 28B + N*12B, where N = number of light samples (1 or more).
*/
struct PathState
{
    // TODO: Check with and without 16B alignment and/or bound as root descriptor.
    // It doesn't seem to be clearly faster to align or use root descriptor currently, but need to look at actual code.

    // Path data (20B)
    float pdf;         ///< Sampling pdf at the last path vertex. Note: This field is only valid if MIS is enabled.
    float3 thp;        ///< Current path throughput.
    uint packedNormal; ///< Normal used for NEE at the last path vertex (packed as 2x 16-bit snorms in octahedral mapping). Note: This field
                       ///< is only valid if MIS is enabled.

    // Shadow rays (12B/ray)
    float3 Lr[kLightSamplesPerVertex]; ///< Unoccluded contribution from each shadow ray (xyz).

    // Sample generator (4-16B, typically 8B).
    SampleGenerator sg; ///< Sample generator state.
};

/** Ray parameters for the scatter and shadow ray(s) at each path vertex.

    This is stored in AoS layout separate from the path state to reduce memory traffic,
    as the tracing pass doesn't need to access any of the other path state.
    The struct is currently 20B + N*16B (N = number of shadow rays per vertex).
*/
struct PathRays
{
    // Scatter ray (20B)
    float3 origin; ///< Origin of the shadow/scatter (non-transmission) rays in world space. This is already offset from the surface by a
                   ///< machine-based epsilon.
    float2 packedScatterDir; ///< Scatter ray direction (packed as 2x 32-bit fp32 in octahedral mapping).

    // Shadow rays (16B/ray)
    float4 shadowRay[kLightSamplesPerVertex]; ///< Shadow ray normalized direction (xyz) and distance (w).

    // Accessor functions
    [mutating]
    void setScatterDir(float3 d) { packedScatterDir = ndir_to_oct_snorm(d); }

    float3 getScatterDir() { return oct_to_ndir_snorm(packedScatterDir); }
};

/** Computes the path state index based on dispatch thread ID.
    If the dispatch thread ID is out of bounds, kInvalidIndex is returned.
*/
uint getIdx(uint3 dispatchThreadId)
{
    // We're addressing the path state in 16x2x1 blocks in scanline order,
    // with the sample layers consecutively in memory.
    // TODO: Experiment with different memory layouts and block sizes.
    uint3 blockId = uint3(dispatchThreadId.x >> 4, dispatchThreadId.y >> 1, dispatchThreadId.z);
    if (any(blockId >= kBlockDim))
        return kInvalidIndex;

    uint blockIdx = (blockId.y * kBlockDim.x + blockId.x) * kBlockDim.z + blockId.z;
    uint threadIdx = ((dispatchThreadId.y & 0x1) << 4) | (dispatchThreadId.x & 0xf);
    return (blockIdx << 5) | threadIdx;
}
