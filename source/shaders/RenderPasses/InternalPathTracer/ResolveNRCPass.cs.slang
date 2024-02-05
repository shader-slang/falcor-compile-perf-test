/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/

/** Compute shader injecting the radiance predicted by the neural cache and creating training records.
*/
import ColorType;
import Params;
import NeuralRadianceCache.NeuralRadianceCache;
import StaticParams;
import Utils.Debug.PixelDebug;
import Rendering.Utils.PixelStats;
import NeuralRadianceCache.Params;

#define DEBUG_VERTEX_ARRAY 0

cbuffer CB
{
    PathTracerParams gParams;

    Texture2D<uint> gSampleOffset;                     ///< Output offset into per-sample buffers. Only valid when kSamplesPerPixel == 0.
    RWStructuredBuffer<ColorType> gSampleColor;        ///< Output per-sample color if kSamplesPerPixel != 1.
    RWTexture2D<float4> gOutputColor;                  ///< Output color buffer if kSamplesPerPixel == 1.

    NRC nrc;
    bool kForTraining;
}

float3 getColor(uint2 pixel, uint sampleIndex)
{
    if (kSamplesPerPixel == 1)
    {
        return gOutputColor[pixel].rgb;
    }
    else
    {
        uint idx = gParams.getSampleOffset(pixel, gSampleOffset) + sampleIndex;
        return gSampleColor[idx].get();
    }
}

void setColor(uint2 pixel, uint sampleIndex, float3 color)
{
    if (kSamplesPerPixel == 1)
    {
        // Accumulate color directly to frame buffer.
        gOutputColor[pixel].rgb = color;
    }
    else
    {
        // Accumulate color to per-sample buffer.
        uint idx = gParams.getSampleOffset(pixel, gSampleOffset) + sampleIndex;
        gSampleColor[idx].set(color);
    }
}

void addColor(uint2 pixel, uint sampleIndex, float3 color)
{
    if (kSamplesPerPixel == 1)
    {
        // Accumulate color directly to frame buffer.
        gOutputColor[pixel].rgb += color;
    }
    else
    {
        // Accumulate color to per-sample buffer.
        uint idx = gParams.getSampleOffset(pixel, gSampleOffset) + sampleIndex;
        color += gSampleColor[idx].get();
        gSampleColor[idx].set(color);
    }
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint2 frameDim = gParams.frameDim;
    uint2 pixel = dispatchThreadId.xy;
    if (kForTraining)
    {
        pixel = pixel * nrc.kTrainingStride.xy + nrc.kTrainingOffset.xy;
        if (any(pixel >= frameDim)) return;
    }
    const uint sampleIndex = dispatchThreadId.z;
    printSetPixel(pixel);
    logSetPixel(pixel);

    const uint3 pathOffset = uint3(pixel.x, pixel.y, sampleIndex) % nrc.kTrainingStride;
    const bool isTrainingPath = all(pathOffset == nrc.kTrainingOffset);

    const uint pathIndex = NRC::PathInfo::getIndex(frameDim, pixel, sampleIndex);
    const NRC::PathInfo path = nrc.pathInfo[pathIndex];

    const float3 prefixThroughput = path.getPrefixThroughput();
    const int queryIndex = int(path.getQueryIndex());

    // If we're not a training path, we can just early exit
    if (!isTrainingPath)
    {
#if DEBUG_VERTEX_ARRAY == 0
        if (queryIndex <= nrc.kMaxPathBounces) addColor(pixel, sampleIndex, nrc.query.radiance[path.queryBufferIndex] * prefixThroughput);
#endif
        return;
    } else if (!kForTraining) return;

    const uint baseIndex = NRC::PathVertex::getIndex(nrc.kTrainingResolution, pixel, sampleIndex, 0, nrc.kTrainingStride.xy, nrc.kMaxPathBounces);

    const int lastIndex = int(path.getVertexCount()) - 1;

    float3 accTraining = float3(0.f);
    float3 accRendering = float3(0.f);

    for (int vertexIndex = lastIndex; vertexIndex >= 0; --vertexIndex)
    {
        NRC::PathVertex vertex = NRC::PathVertex::createFromPacked(nrc.pathVertices[baseIndex + vertexIndex]);
        float3 overallAlbedo = nrc.kLearnIrradiance ? (vertex.albedo + vertex.specular) : float3(1.0f);

        if (isTrainingPath)
        {
            if (vertexIndex == lastIndex && path.queryBufferIndexTraining != 0xFFFFFFFF)
            {
                accTraining = nrc.query.radiance[path.queryBufferIndexTraining] * overallAlbedo + (nrc.kRadianceCacheDirect ? 0 : vertex.radiance);
            }
            else
            {
                accTraining *= vertex.throughput;
                // if (vertexIndex >= queryIndex)
                {
                    uint recordIndex = nrc.counters.increment((uint)NRCCounters::kTrainingRecords);
                    nrc.training.radianceParams[recordIndex] = NRC::RadianceParams::createFrom(vertex);
                    float3 target = min(accTraining + (nrc.kRadianceCacheDirect ? vertex.radiance : 0), nrc.training.kMaxRadianceValue);
                    nrc.training.radiance[recordIndex] = target;
                }
                accTraining += vertex.radiance;
            }
        }

#if DEBUG_VERTEX_ARRAY
        accRendering *= vertex.throughput;
        accRendering += vertex.radiance;
#else
        if (vertexIndex > queryIndex) continue;
        accRendering *= vertex.throughput;

        // Slightly hacky. Should ideally have the path depth of the vertex available
        // (instead of vertexIndex), but it'd occupy extra memory bandwidth
        // that we don't want to waste for debug visualization purposes
        accRendering += (vertexIndex == queryIndex) ? (nrc.query.radiance[path.queryBufferIndex] * overallAlbedo + (nrc.kRadianceCacheDirect ? 0 : vertex.radiance)) : vertex.radiance;
#endif
    }
    accRendering *= prefixThroughput;
#if DEBUG_VERTEX_ARRAY
        accRendering *= nrc.kTrainingStride.x * nrc.kTrainingStride.y;
#endif
    addColor(pixel, sampleIndex, accRendering);
}
