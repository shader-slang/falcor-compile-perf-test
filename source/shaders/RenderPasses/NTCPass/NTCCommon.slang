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

import Utils.Sampling.TinyUniformSampleGenerator;
import Scene.Scene;
import Scene.ShadingData;
import Rendering.Lights.LightHelpers;
import RenderPasses.NTCPass.BRDF;

__exported import Internal.Utils.Neural.TIN.TinCommon;

float2 boxMullerTransform(float2 jitter)
{
    float2 r;
    let mag = 0.4 * sqrt(-2.0 * log(jitter.x));
    r.x = mag * cos(2.0 * M_PI * jitter.y);
    r.y = mag * sin(2.0 * M_PI * jitter.y);
    r = clamp(r, -1.0f, 1.0f);
    return r;
}

float2 jitterUVBilinear(float2 fract_part, float rand)
{
    float4 ws = float4(
        (1.0 - fract_part.x) * (1.0 - fract_part.y),
        fract_part.x * (1.0 - fract_part.y),
        (1.0 - fract_part.x) * fract_part.y,
        fract_part.x * fract_part.y
    );
    ws = float4(ws.x, ws.x + ws.y, ws.x + ws.y + ws.z, ws.x + ws.y + ws.z + ws.w);
    // Instead of dividing CDF by the maximum, multiply the comparand.
    rand *= ws.w;

    int sel = 3;
    if (rand < ws.x)
    {
        sel = 0;
    }
    else if (rand < ws.y)
    {
        sel = 1;
    }
    else if (rand < ws.z)
    {
        sel = 2;
    }
    float dx = (sel == 1 || sel == 3) ? 1.0 : 0.0;
    float dy = (sel == 2 || sel == 3) ? 1.0 : 0.0;
    return float2(dx, dy);
}

float2 jitterUVGaussian(float2 fract_part, float rand)
{
#define EXTENT 4
#define NEG_RANGE ((EXTENT - 1) / 2)
#define POS_RANGE (EXTENT - NEG_RANGE)
#define W_COUNT (EXTENT * EXTENT)

    float2 offsets[W_COUNT];
    int it = 0;
    for (int x = -NEG_RANGE; x < POS_RANGE; ++x)
    {
        for (int y = -NEG_RANGE; y < POS_RANGE; ++y)
        {
            offsets[it++] = float2(x, y);
        }
    }

    // The whole following code can be optimized by rewriting in a binary-tree fashion.
    // This would be an optimization only, not changing the functionality.
    float gaussian_weights_cdf[W_COUNT];
    float prev_val = 0.0f;
    for (int i = 0; i < W_COUNT; ++i)
    {
        float2 off = offsets[i] - fract_part;
        float dist_sq = off.x * off.x + off.y * off.y;
        float inv_sigma_sq = 5.0;
        float w = exp2(-0.5 * dist_sq * inv_sigma_sq);
        gaussian_weights_cdf[i] = prev_val + w;
        prev_val = gaussian_weights_cdf[i];
    }

    // Instead of dividing CDF by the maximum, multiply the comparand.
    rand *= prev_val;
    for (int i = 0; i < W_COUNT - 1; ++i)
    {
        if (rand < gaussian_weights_cdf[i])
            return offsets[i];
    }
    return offsets[W_COUNT - 1];
}

float2 jitterUV(const uint uvJitterMode, float2 uv, float2 dims, float rand)
{
    float2 uv_full = uv * dims - 0.5;
    float2 left_top = floor(uv_full);
    int2 left_top_i = int2(uv_full);
    float2 fract_part = uv_full - left_top;

    float2 offset = uvJitterMode == 1 ? jitterUVBilinear(fract_part, rand) : jitterUVGaussian(fract_part, rand);

    return (left_top + offset + 0.5) / dims;
}

float2 getJitter(TinyUniformSampleGenerator sg, uint uvJitterMode)
{
    if (uvJitterMode == 0)
        return float2(0);
    let jitter = sampleNext2D(sg);
    return uvJitterMode == 1 ? jitter - 0.5f : boxMullerTransform(jitter);
}

float2 roundToPixel(float2 uv, float2 jitter, float2 dims)
{
    // Add jitter and round to pixel location.
    float2 xy = uv * dims + jitter;
    xy = floor(xy) + 0.5;
    float2 ret = xy / dims;
    return ret;
}

// Hand-crafted to empirically match HW texture LOD function.
float computeLod(uint2 dim, float4 textureGrads, float minLod, float maxLod, float lodJitter)
{
    // we subtract 2 LOD levels from maxLOD, as we cannot go below 4x4 texels, due to BC compression.
    let dUVdx = textureGrads.xy;
    let dUVdy = textureGrads.zw;
    let maxDelta = max(dim.x * max(abs(dUVdx.x), abs(dUVdy.x)), dim.y * max(abs(dUVdx.y), abs(dUVdy.y)));
    return min(maxLod, max(minLod, log2(maxDelta) + lodJitter));
}

uint lod2NeuralLod(uint lod, uint minNeuralLod, uint maxNeuralLod)
{
    float neuralLod = 0;
    if (lod >= 4)
        neuralLod = 1 + ((lod - 4) / 2);
    return min(maxNeuralLod, max(minNeuralLod, neuralLod));
}

float normalizeLod(int lod, float maxLod)
{
    return lod / (1 + maxLod);
}

float4 shadeSample(float3 albedo, float3 normal, float ao, float roughness, float metallic, float specular, ShadingData sd)
{
    AnalyticLightSample ls;
    evalLightApproximate(sd.posW, gScene.getLight(0), ls);

    normal = normalize(normal * 2 - 1);
    float3 s_normal = sd.frame.N * normal.z + sd.frame.T * normal.x + sd.frame.B * normal.y;
    float NdotL = dot(s_normal, ls.dir);
    float3 lighting =
        BRDF(ls.dir, sd.V, s_normal, albedo, roughness, metallic, specular) * ls.Li * NdotL + 0.2 * (1 - metallic) * ao * albedo;

    return float4(lighting, 1);
}

namespace Tin
{
    struct FeatureGrid<let WIDTH : uint, let HEIGHT : uint, let NUM_CH : uint, let NUM_BITS : uint, let ALL_CORNERS : bool>
    {
        half unpack(uint packed, uint bit_offset)
        {
            const uint NUM_BINS = 1 << NUM_BITS;
            const uint MASK = NUM_BINS - 1;
            const float STEP = 1 / (NUM_BINS / 2 + 0.5f);

            uint xu = (packed >> bit_offset) & MASK;
            int xi = (int)xu - (NUM_BINS / 2 - 1);
            float xf = xi * STEP;
            return xf;
        }

        half2 blend(half2 x0, half2 x1, half a) { return x0 * (1.h - a) + x1 * a; }

        uint getBufferOffset(uint scale)
        {
            const uint NUM_PACKED_UINT8 = 8 / NUM_BITS;
            const uint NUM_CHUNKS = NUM_CH / NUM_PACKED_UINT8;
            uint width = WIDTH / scale;
            uint height = HEIGHT / scale;
            return width * height * NUM_CHUNKS;
        }

        void sample<let MAT_CH : uint>(
            float2 uv,
            uint scale,
            uint bufferOffset,
            uint slotOffset,
            Buffer<uint> featureGrids,
            inout HPackedArray<MAT_CH> res,
            int sel_feature
        )
        {
            uint width = WIDTH / scale;
            uint height = HEIGHT / scale;

            float2 xy = uv * float2(width, height) - 0.5f;
            float2 xyBase = floor(xy);
            float2 alpha = xy - xyBase;

            const uint NUM_PACKED_UINT8 = 8 / NUM_BITS;
            const uint NUM_CHUNKS = NUM_CH / NUM_PACKED_UINT8;

            float x0 = max(0.f, xyBase.x);
            float y0 = max(0.f, xyBase.y);
            float x1 = min(width - 1, xyBase.x + 1);
            float y1 = min(height - 1, xyBase.y + 1);

            uint baseIndex00 = (y0 * width + x0) * NUM_CHUNKS + bufferOffset;
            uint baseIndex01 = (y0 * width + x1) * NUM_CHUNKS + bufferOffset;
            uint baseIndex11 = (y1 * width + x1) * NUM_CHUNKS + bufferOffset;
            uint baseIndex10 = (y1 * width + x0) * NUM_CHUNKS + bufferOffset;

            [unroll]
            for (uint i = 0; i < NUM_CHUNKS; i++)
            {
                uint chunk00 = featureGrids.Load(baseIndex00 + i);
                uint chunk01 = featureGrids.Load(baseIndex01 + i);
                uint chunk11 = featureGrids.Load(baseIndex11 + i);
                uint chunk10 = featureGrids.Load(baseIndex10 + i);

                [unroll]
                for (uint k = 0; k < NUM_PACKED_UINT8; k += 2)
                {
                    half2 d00 = 0, d01 = 0, d11 = 0, d10 = 0;

                    uint feat_idx = (i * NUM_PACKED_UINT8 + k);

                    if (sel_feature < 0 || feat_idx == sel_feature)
                    {
                        d00.x = unpack(chunk00, k * NUM_BITS);
                        d01.x = unpack(chunk01, k * NUM_BITS);
                        d11.x = unpack(chunk11, k * NUM_BITS);
                        d10.x = unpack(chunk10, k * NUM_BITS);

                        d00.y = unpack(chunk00, (k + 1) * NUM_BITS);
                        d01.y = unpack(chunk01, (k + 1) * NUM_BITS);
                        d11.y = unpack(chunk11, (k + 1) * NUM_BITS);
                        d10.y = unpack(chunk10, (k + 1) * NUM_BITS);
                    }

                    uint sel_pair = feat_idx / 2 + slotOffset;

                    if (ALL_CORNERS)
                    {
                        res.set_packed_item(sel_pair, half2::pack(d00));
                        res.set_packed_item(sel_pair + 1 * NUM_CH / 2, half2::pack(d10));
                        res.set_packed_item(sel_pair + 2 * NUM_CH / 2, half2::pack(d01));
                        res.set_packed_item(sel_pair + 3 * NUM_CH / 2, half2::pack(d11));
                    }
                    else
                    {
                        half2 d = blend(blend(d00, d01, alpha.x), blend(d10, d11, alpha.x), alpha.y);
                        res.set_packed_item(sel_pair, half2::pack(d));
                    }
                }
            }
        }
    }
}
