/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"
__exported import StaticParams;

struct NRCData
{
    /** Path flags. The path flags are currently stored in a uint16_t.
    */
    enum class PathFlags
    {
        forTraining = 0x0001,   ///< training paths should not be terminated before we reach kMaxBounces
        exitedScene = 0x0002,   ///< training paths that exited the scene should not be "Q-learned"
        unbiased    = 0x0004,   ///< some of the training paths are marked as "unbiased" to be extended through the entire scene

        // Bit 3-15 are still unused.
    };

    uint16_t    flags;                      ///< Flags indicating the current status. This can be multiple PathFlags flags OR'ed together.
    float3      trainingVertexRadiance;
    float3      trainingVertexThroughput;
    uint16_t    vertexCount;                ///< The number of vertices processed.
    uint16_t    queryIndex;                 ///< The index of the vertex at which we will query the cache.
    float16_t   primarySpreadRadius;        ///< Approximated as `d^2 / cos` at primary hit.
    float16_t   cumulSpreadRadius;          ///< Square root of the cumulative area spread at the current path vertex.
    uint        queryBufferIndex;
    uint        queryBufferIndexTraining;

    bool isForTraining() { return hasFlag(PathFlags::forTraining); }
    bool hasExitedScene() { return hasFlag(PathFlags::exitedScene); }
    bool isUnbiased() { return hasFlag(PathFlags::unbiased); }
    [mutating] void setForTraining(bool value = true) { setFlag(PathFlags::forTraining, value); }
    [mutating] void setExitedScene(bool value = true) { setFlag(PathFlags::exitedScene, value); }
    [mutating] void setUnbiased(bool value = true) { setFlag(PathFlags::unbiased, value); }
    // We reuse the trainingVertexRadiance field to hold the path prefix throughput
    // for non-training paths (instead of adding a separate field) for performance.
    [mutating] void setPathPrefixThroughput(const float3 thp) { trainingVertexRadiance = thp; }
    float3 getPathPrefixThroughput() { return trainingVertexRadiance; }

    bool hasFlag(PathFlags flag) { return (flags & uint(flag)) != 0; }
    [mutating] void setFlag(PathFlags flag, bool value = true)
    {
        if (value) flags |= uint16_t(uint(flag));
        else flags &= ~uint16_t(uint(flag));
    }

    [mutating] void init(uint maxPathBounces)
    {
        flags = 0;
        vertexCount = 0;
        queryIndex = maxPathBounces + 1;    // Out of reach by default.
        cumulSpreadRadius = 0.f;
        queryBufferIndex = 0xFFFFFFFF;
        queryBufferIndexTraining = 0xFFFFFFFF;
    }

    bool isQueryVertex()
    {
        return vertexCount - 1 == queryIndex;
    }

    bool isHeuristicSatisfied(float threshold)
    {
        return primarySpreadRadius > 0.f && cumulSpreadRadius > threshold * primarySpreadRadius;
    }

    /** Update the path spread. Used by NRC to drive the termintation heuristic.
        \param[in] vertexIndex Index of current vertex along the path.
        \param[in] distance. Distance since last path vertex.
        \param[in] view. View direction at current vertex.
        \param[in] normal. Surface normal at current vertex.
        \param[in] pdf. PDF of the current vertex BSDF sample.
        \param[in] normal. Indicate if we sampled a delta lobe for the current vertex.
    */
    [mutating] void updatePathSpread(const uint vertexIndex, const float distance, const float3 view, const float3 normal, const float pdf, const bool isDelta)
    {
        float cosGamma = abs(dot(view, normal));
        if (vertexIndex == 1)
        {
            primarySpreadRadius = (float16_t)(distance / sqrt(cosGamma * M_1_4PI));
        }
        else if (!isDelta)
        {
            cumulSpreadRadius += (float16_t)(distance / sqrt(cosGamma * pdf));
        }
    }

};
