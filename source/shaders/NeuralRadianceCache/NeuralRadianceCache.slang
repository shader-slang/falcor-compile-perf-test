/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
import Utils.Debug.PixelDebug;
import Utils.Math.FormatConversion;
import Utils.Math.MathHelpers;
import Utils.Math.PackedFormats;
import RenderPasses.InternalPathTracer.PathState;
import Scene.ShadingData;
import Scene.Material.MaterialTypes;
import Rendering.Materials.IMaterialInstance;

__exported import NeuralRadianceCache.Params;

#define NRC_COMPACT_PATH_INFO 1
#define NRC_COMPACT_PATH_VERTEX 1

float2 safe_cartesian_to_spherical_unorm(float3 p)
{
    if (any(!isfinite(p))) return float2(0.0f);
    return cartesian_to_spherical_unorm(p);
}

/** Neural radiance cache.

    This exposes the data structures needed to query and train the neural cache.
*/
struct NRC
{
#if NRC_COMPACT_PATH_VERTEX
    struct PackedPathVertex
    {
        uint data[8];

        [mutating] void init(const ShadingData sd, const BSDFProperties bsdfProperties, const float3 pos)
        {
            // data[0] = encodeLogLuvHDR(float3(0.f));
            // data[1] = encodeLogLuvHDR(float3(1.f));
            data[2] = packUnorm2x16(pos.xy);
            data[3] = packUnorm2x16({ pos.z, bsdfProperties.roughness });
            data[4] = encodeNormal2x16(any(isnan(sd.frame.N)) ? sd.faceN : sd.frame.N);
            data[5] = encodeNormal2x16(sd.V);
            data[6] = packR11G11B10(bsdfProperties.diffuseReflectionAlbedo);
            data[7] = packR11G11B10(bsdfProperties.specularReflectionAlbedo);
        }

        [mutating] void init(const float3 view, const float3 pos)
        {
            // data[0] = encodeLogLuvHDR(float3(0.f));
            // data[1] = encodeLogLuvHDR(float3(1.f));
            data[2] = packUnorm2x16(pos.xy);
            data[3] = packUnorm2x16({ pos.z, 0.f });
            data[4] = encodeNormal2x16(view);
            data[5] = encodeNormal2x16(view);
            data[6] = packR11G11B10(0.5f);
            data[7] = packR11G11B10(0.5f);
        }

        [mutating] void update(const float3 radiance, const float3 throughput)
        {
            data[0] = encodeLogLuvHDR(radiance);
            data[1] = encodeLogLuvHDR(throughput);
        }
    };
#else
    struct PackedPathVertex
    {
        float3 radiance;        ///< Reflected radiance
        float3 throughput;      ///< Throughput to the next vertex

        float3 position;        ///< World space position
        float  linearRoughness; ///< Material roughness

        float3 normal;          ///< Sampled direction
        float3 viewDirection;   ///< Direction towards the previous path vertex

        float3 albedo;          ///< Base diffuse reflectance
        float3 specular;        ///< Base specular reflectance

        [mutating] void init(const ShadingData sd, const BSDFProperties bsdfProperties, const float3 pos)
        {
            // radiance = float3(0.f);
            // throughput = float3(1.f);
            position = pos;
            linearRoughness = bsdfProperties.roughness;
            normal = any(isnan(sd.frame.N)) ? sd.faceN : sd.frame.N;
            viewDirection = sd.V;
            albedo = bsdfProperties.diffuseReflectionAlbedo * (1.0f - bsdfProperties.specularTransmissionAlbedo);
            specular = bsdfProperties.specularReflectionAlbedo;
        }

        [mutating] void init(const float3 view, const float3 pos)
        {
            // radiance = float3(0.f);
            // throughput = float3(1.f);
            position = pos;
            linearRoughness = 0.f;
            normal = view;
            viewDirection = view;
            albedo = 0.5f;
            specular = 0.5f;
        }

        [mutating] void update(const float3 rad, const float3 thp)
        {
            radiance = rad;
            throughput = thp;
        }
    };
#endif

    /** Struct holding path vertex data for training the Neural Radiance Cache.
    */
    struct PathVertex
    {
        float3 radiance;        ///< Reflected radiance
        float3 throughput;      ///< Throughput to the next vertex

        float3 position;        ///< World space position
        float  linearRoughness; ///< Material roughness

        float3 normal;          ///< Sampled direction
        float3 viewDirection;   ///< Direction towards the previous path vertex

        float3 albedo;          ///< Base diffuse reflectance
        float3 specular;        ///< Base specular reflectance

        static PathVertex createFromPacked(const PackedPathVertex packed)
        {
            PathVertex vertex;
#if NRC_COMPACT_PATH_VERTEX
            vertex.radiance = decodeLogLuvHDR(packed.data[0]);
            vertex.throughput = decodeLogLuvHDR(packed.data[1]);
            vertex.position.xy = unpackUnorm2x16(packed.data[2]);
            float2 tmp = unpackUnorm2x16(packed.data[3]);
            vertex.position.z = tmp.x;
            vertex.linearRoughness = tmp.y;
            vertex.normal = decodeNormal2x16(packed.data[4]);
            vertex.viewDirection = decodeNormal2x16(packed.data[5]);
            vertex.albedo = unpackR11G11B10(packed.data[6]);
            vertex.specular = unpackR11G11B10(packed.data[7]);
#else
            vertex.radiance = packed.radiance;
            vertex.throughput = packed.throughput;
            vertex.position = packed.position;
            vertex.linearRoughness = packed.linearRoughness;
            vertex.normal = packed.normal;
            vertex.viewDirection = packed.viewDirection;
            vertex.albedo = packed.albedo;
            vertex.specular = packed.specular;
#endif
            return vertex;
        }

        static PathVertex createFromPacked(const PackedPathVertex packed, const float3 radiance, const float3 throughput)
        {
            PathVertex vertex;
            vertex.radiance = radiance;
            vertex.throughput = throughput;
#if NRC_COMPACT_PATH_VERTEX
            vertex.position.xy = unpackUnorm2x16(packed.data[2]);
            float2 tmp = unpackUnorm2x16(packed.data[3]);
            vertex.position.z = tmp.x;
            vertex.linearRoughness = tmp.y;
            vertex.normal = decodeNormal2x16(packed.data[4]);
            vertex.viewDirection = decodeNormal2x16(packed.data[5]);
            vertex.albedo = unpackR11G11B10(packed.data[6]);
            vertex.specular = unpackR11G11B10(packed.data[7]);
#else
            vertex.position = packed.position;
            vertex.linearRoughness = packed.linearRoughness;
            vertex.normal = packed.normal;
            vertex.viewDirection = packed.viewDirection;
            vertex.albedo = packed.albedo;
            vertex.specular = packed.specular;
#endif
            return vertex;
        }

        static uint getIndex(uint2 frame, uint2 pixel, const uint sampleIndex, const uint length, const uint2 trainingStride, const uint maxPathBounces)
        {
            uint index = length;
            uint stride = 1 + maxPathBounces;

            pixel /= trainingStride;

            // Index in a linearized Tensor: (height, width, spp)
            index += sampleIndex * stride;
            stride *= kSamplesPerPixel;
            index += pixel.x * stride;
            stride *= frame.x;
            index += pixel.y * stride;
            return index;
        }
    };

    struct RadianceParams
    {
        float3 position;
        float roughness;
        float2 normal;
        float2 viewDirection;

        float3 albedo;
        float3 specular;

        static RadianceParams createFrom(const ShadingData sd, const BSDFProperties bsdfProperties, const float3 pos)
        {
            RadianceParams params;
            params.position = pos;
            params.roughness = bsdfProperties.roughness;
            params.normal = safe_cartesian_to_spherical_unorm(any(isnan(sd.frame.N)) ? sd.faceN : sd.frame.N);
            params.viewDirection = safe_cartesian_to_spherical_unorm(sd.V);
            params.albedo = bsdfProperties.diffuseReflectionAlbedo;
            params.specular = bsdfProperties.specularReflectionAlbedo;
            return params;
        }

        static RadianceParams createFrom(const float3 view, const float3 pos)
        {
            RadianceParams params;
            params.position = pos;
            params.roughness = 0.f;
            params.normal = safe_cartesian_to_spherical_unorm(view);
            params.viewDirection = safe_cartesian_to_spherical_unorm(view);
            params.albedo = 0.5f;
            params.specular = 0.5f;
            return params;
        }

        static RadianceParams createFrom(const PathVertex vertex)
        {
            RadianceParams params;
            params.position = vertex.position;
            params.roughness = vertex.linearRoughness;
            params.normal = safe_cartesian_to_spherical_unorm(vertex.normal);
            params.viewDirection = safe_cartesian_to_spherical_unorm(vertex.viewDirection);
            params.albedo = vertex.albedo;
            params.specular = vertex.specular;
            return params;
        }
    };

    enum class PathInfoFlags
    {
        exitedScene = 0x0001,
    };

    static const uint kVertexCountOffset = 0;
    static const uint kQueryIndexOffset = 8;
    static const uint kFlagsOffset = 16;
    static const uint kEightBitsMask = (1 << 8) - 1;

    struct PathInfo
    {
        uint flagsAndIndices;
    #if NRC_COMPACT_PATH_INFO
        uint prefixThroughput;
    #else
        float3 prefixThroughput;
    #endif
        uint queryBufferIndex;
        uint queryBufferIndexTraining;

        static PathInfo createFromPathState(const PathState path)
        {
            PathInfo pathInfo;
            float3 thp = path.nrcData.isForTraining() ? path.thp : path.nrcData.getPathPrefixThroughput();
            pathInfo.flagsAndIndices = uint(path.nrcData.vertexCount) << kVertexCountOffset |
                                       uint(path.nrcData.queryIndex) << kQueryIndexOffset;
            pathInfo.setFlag(PathInfoFlags::exitedScene, path.nrcData.hasExitedScene());
    #if NRC_COMPACT_PATH_INFO
            pathInfo.prefixThroughput = encodeLogLuvHDR(max(0, thp));
    #else
            pathInfo.prefixThroughput = max(0, thp);
    #endif
            pathInfo.queryBufferIndex = path.nrcData.queryBufferIndex;
            pathInfo.queryBufferIndexTraining = path.nrcData.queryBufferIndexTraining;
            return pathInfo;
        }

        bool hasFlag(PathInfoFlags flag) { return ((flagsAndIndices >> kFlagsOffset) & uint(flag)) != 0; }
        [mutating] void setFlag(PathInfoFlags flag, bool value = true)
        {
            if (value) flagsAndIndices |= (uint(flag) << kFlagsOffset);
            else flagsAndIndices &= ~(uint(flag) << kFlagsOffset);
        }

        [mutating] void setAsBackground(const float3 color)
        {
            flagsAndIndices = 0;
            setFlag(PathInfoFlags::exitedScene, true);
    #if NRC_COMPACT_PATH_INFO
            prefixThroughput = encodeLogLuvHDR(float3(0.f));
    #else
            prefixThroughput = float3(0.f);
    #endif
        }

        uint getQueryIndex()
        {
            return (flagsAndIndices >> kQueryIndexOffset) & kEightBitsMask;
        }

        uint getVertexCount()
        {
            return (flagsAndIndices >> kVertexCountOffset) & kEightBitsMask;
        }

        float3 getPrefixThroughput()
        {
    #if NRC_COMPACT_PATH_INFO
            return decodeLogLuvHDR(prefixThroughput);
    #else
            return prefixThroughput;
    #endif
        }

        bool hasExitedScene()
        {
            return hasFlag(PathInfoFlags::exitedScene);
        }

        static uint getIndex(const uint2 frame, const uint2 pixel, const uint sampleIndex)
        {
            // Index in a linearized Tensor: (height, width, spp, vertices)
            uint index = sampleIndex;
            uint stride = kSamplesPerPixel;
            index += pixel.x * stride;
            stride *= frame.x;
            index += pixel.y * stride;
            return index;
        }
    };

    struct Counters
    {
        RWByteAddressBuffer data;
        uint increment(uint counterID)
        {
            uint laneCount = WaveActiveCountBits(true);
            uint laneOffset = WavePrefixCountBits(true);
            uint originalValue;
            if (WaveIsFirstLane())
            {
                data.InterlockedAdd(counterID * 4, laneCount, originalValue);
            }
            originalValue = WaveReadLaneFirst(originalValue); // Broadcast to all active threads
            return originalValue + laneOffset;
        }
    };

    const bool kRadianceCacheDirect;
    const bool kLearnIrradiance;
    const bool kSkipDeltaVertices;
    const bool kUseTerminationHeuristic;
    const float kTerminationHeuristicThreshold;
    const uint kMaxPathBounces;                         ///< Max path length
    const uint kQueryVertexIndex;                       ///< Vertex where cache is queried.
    const bool kForceEarlyQueryHairFur;                 ///< If enabled, `kMaxQueryVertexIndexHairFur` will be enforced when using the termination heuristic.
    const uint kMaxQueryVertexIndexHairFur;             ///< Can be used to bypass termination heuristic of hair/fur.
    const uint3 kTrainingStride;                        ///< Defines a training block size, keep one path per block.
    const uint3 kTrainingOffset;                        ///< Offset of the path used for training within the block.
    const uint2 kTrainingResolution;
    RWStructuredBuffer<PathInfo> pathInfo;              ///< Misc path info (vertexCount, queryIndex).
    RWStructuredBuffer<PackedPathVertex> pathVertices;  ///< Path vertex data used to train the neural radiance cache.
    Counters counters;

    struct Query {
        RWStructuredBuffer<float3> radiance;
        RWStructuredBuffer<RadianceParams> radianceParams;
    } query;

    struct Training {
        const float kMaxRadianceValue;                  ///< Used to clamp target value in training records; this mitigates divergence due to fireflies.
        RWStructuredBuffer<float3> radiance;
        RWStructuredBuffer<RadianceParams> radianceParams;
    } training;

    const float3 kSceneMinPoint;
    const float kSceneExtentMax;

    /** Update the current vertex information (if any) and initialize a new vertex. All data is stored in a separate vertex array.
        \param[in,out] path Path.
        \param[in] sd. Shading data of current path vertex.
        \param[in] isDeltaLobe. Indicate whether we sample a delta lobe on the new vertex.
        \param[in] isVolumeHit. Create volume vertex entry if true, and a surface vertex entry otherwise.
        \return Returns true if a query was created and the path can be terminated.
    */
    bool createQuery(inout PathState path, const ShadingData sd, const BSDFProperties bsdfProperties, const bool isDeltaLobe, const bool isVolumeHit)
    {
        const bool skipVertex = (kSkipDeltaVertices || kUseTerminationHeuristic) && isDeltaLobe;
        if (!skipVertex)
        {
            // Check if we can query the cache at the current vertex.
            bool isSafeForQuery = false;
            if (path.nrcData.queryIndex > kMaxPathBounces) // Check that query index is not already set (index is initialized at `kMaxPathBounces + 1`).
            {
                if (!kUseTerminationHeuristic)
                {
                    isSafeForQuery = kQueryVertexIndex <= path.nrcData.vertexCount;
                }
                else
                {
                    isSafeForQuery = path.nrcData.isHeuristicSatisfied(kTerminationHeuristicThreshold);
                    if (kForceEarlyQueryHairFur && sd.mtl.getMaterialType() == MaterialType::Hair)
                    {
                        isSafeForQuery |= path.nrcData.vertexCount >= kMaxQueryVertexIndexHairFur;
                    }
                }
            }

            // Finalize previous vertex
            if (path.nrcData.isForTraining() && path.nrcData.vertexCount > 0)
            {
                const uint arrayIndex = PathVertex::getIndex(kTrainingResolution, path.getPixel(), path.getSampleIdx(), path.nrcData.vertexCount - 1, kTrainingStride.xy, kMaxPathBounces);
                pathVertices[arrayIndex].update(path.nrcData.trainingVertexRadiance, path.nrcData.trainingVertexThroughput);
            }

            // Always update vertex count, even for non-training paths. The path.nrcData.vertexCount variable
            // mostly mirrors path.getVertexIndex(), but it does not count specular vertices if these were
            // marked to be skipped. This is needed to ensure that a surface scene in a mirror is handled similarly
            // to surfaces seen directly.
            path.nrcData.vertexCount += uint16_t(1);

            // Store path vertex
            if (path.nrcData.isForTraining())
            {
                path.nrcData.trainingVertexRadiance = float3(0.f);
                path.nrcData.trainingVertexThroughput = float3(1.f);
                const uint arrayIndex = PathVertex::getIndex(kTrainingResolution, path.getPixel(), path.getSampleIdx(), path.nrcData.vertexCount - 1, kTrainingStride.xy, kMaxPathBounces);
                if (isVolumeHit)
                {
                    pathVertices[arrayIndex].init(-path.dir, (path.origin - kSceneMinPoint) / kSceneExtentMax);
                }
                else
                {
                    pathVertices[arrayIndex].init(sd, bsdfProperties, (sd.posW - kSceneMinPoint) / kSceneExtentMax);
                }
            }

            // Create query record
            if (isSafeForQuery)
            {
                path.nrcData.queryIndex = path.nrcData.vertexCount - 1;
                if (path.nrcData.isForTraining())
                {
                    path.nrcData.cumulSpreadRadius = 0; // Reset the heuristic; it'll be re-used to terminate training paths
                }
                else
                {
                    if (kLearnIrradiance && !isVolumeHit) path.nrcData.setPathPrefixThroughput(path.thp * (bsdfProperties.diffuseReflectionAlbedo + bsdfProperties.specularReflectionAlbedo));
                    else path.nrcData.setPathPrefixThroughput(path.thp);
                }

                path.nrcData.queryBufferIndex = counters.increment((uint)NRCCounters::kQueries);
                if (isVolumeHit)
                {
                    query.radianceParams[path.nrcData.queryBufferIndex] = RadianceParams::createFrom(-path.dir, (path.origin - kSceneMinPoint) / kSceneExtentMax);
                }
                else
                {
                    query.radianceParams[path.nrcData.queryBufferIndex] = RadianceParams::createFrom(sd, bsdfProperties, (sd.posW - kSceneMinPoint) / kSceneExtentMax);
                }
            }

            // Terminate early if the cache already includes direct reflected radiance.
            // Otherwise, we will terminate later, after NEE and the scatter ray has been computed.
            if (kRadianceCacheDirect) {
                bool terminateShortPath = !path.nrcData.isForTraining() && path.nrcData.isQueryVertex();
                bool terminateTrainingPath = path.getVertexIndex() == kMaxPathBounces || !path.nrcData.isUnbiased() && path.nrcData.vertexCount > path.nrcData.queryIndex && path.nrcData.isHeuristicSatisfied(kTerminationHeuristicThreshold);
                if (terminateShortPath || terminateTrainingPath) return true;
            }
        }

        return false;
    }
};
