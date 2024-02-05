/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
import Internal.Scene2.SharedTypes;
import Internal.Scene2.Cameras.Camera;
import Internal.Scene2.Scene2;
import Internal.Scene2.HitInfo;
import Internal.Scene2.Geometry.GeoBase;
import Internal.Scene2.VertexData;
import Internal.Tools.Scene2Test.SlangRenderer.ShaderRtDelegate;

import Utils.Math.HashUtils;
import Utils.Math.BitTricks;
import Utils.Math.Ray;

import Rendering.Lights.EnvMapSampler;

ShaderRtDelegate gShaderRtDelegate;
#define SCENE2_GEO_RT_DELEGATE ShaderRtDelegate
#define SCENE2_GEO_RT_DELEGATE_INSTANCE gShaderRtDelegate
#include "Internal/Scene2/Geometry/GeoRtEntryPoints.slangh"

ParameterBlock<EnvMapSampler> gEnvMapSampler;
ParameterBlock<RenderUniforms> uniforms;
RWTexture2D<float4> renderTexture;

#if defined(USE_SER)
static const uint kUseSER = USE_SER;
#else
static const uint kUseSER = 1;
#endif

[shader("miss")]
void scatterMiss(inout SCENE2_GEO_RT_DELEGATE::Payload payload: SV_RayPayload)
{
    SCENE2_GEO_RT_DELEGATE_INSTANCE.handleMiss(payload);
}

[shader("miss")]
void visibilityMiss(inout SCENE2_GEO_RT_DELEGATE::VisibilityPayload payload: SV_RayPayload)
{
    payload.visible = true;
}

[shader("raygeneration")]
void rayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 frameDim = DispatchRaysDimensions().xy;
    if (all(pixel >= frameDim))
        return;

    if (SCENE2_GEO_RT_DELEGATE_INSTANCE.isInline())
    {
        renderTexture[pixel] = float4(1, 0, 0, 1.f);
        return;
    }

    float2 uv = (float2(pixel) / float2(uniforms.frameDim) - 0.5f) * 2.f;

    Ray ray = gScene.getCamera().computeRayPinhole(pixel, uniforms.frameDim, false);

    float3 color = float3(0.f);
    const uint sampleCount = uniforms.sampleCount;
    for (uint sampleNumber = 0; sampleNumber < sampleCount; ++sampleNumber)
    {
        uint seed = blockCipherTEA(interleave_32bit(pixel), sampleNumber + uniforms.frameIndex * sampleCount).x;

        if (kUseSER != 0)
            color += SCENE2_GEO_RT_DELEGATE_INSTANCE.tracePrimarySER(ray, seed);
        else
            color += SCENE2_GEO_RT_DELEGATE_INSTANCE.tracePrimary(ray, seed);
    }
    color *= (1.f / sampleCount);

    renderTexture[pixel] = float4(color, 1.f);
}
