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
import Internal.Tools.Scene2Test.SlangRenderer.Renderer;

import Utils.Math.HashUtils;
import Utils.Math.BitTricks;
import Utils.Math.Ray;

ParameterBlock<RenderUniforms> uniforms;
RWTexture2D<float4> renderTexture;

[shader("compute")]
[numthreads(16, 16, 1)]
void renderMain(uint3 threadIdx: SV_DispatchThreadID)
{
    const uint2 pixel = threadIdx.xy;
    if (any(pixel >= uniforms.frameDim))
        return;

    if (!gRendererInline.isInline())
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
        color += gRendererInline.tracePrimary(ray, seed);
    }
    color *= (1.f / sampleCount);

    renderTexture[pixel] = float4(color, 1.f);
}
