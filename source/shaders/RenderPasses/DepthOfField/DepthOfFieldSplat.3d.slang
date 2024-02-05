/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
import DepthOfFieldSplat;

struct VertexToPixel
{
    float4 position : SV_Position;
    nointerpolation float4 color : COLOR;
    noperspective float2 xy : DISTANCE;
    nointerpolation float radius : RADIUS;
};

cbuffer gDoFCommonParams
{
    uint2 inputSize;
    float2 invInputSize;
    float cameraAnisotropy;
    float subjectDistance;
    float lensScale;
    float splatComputeMaxRadius;
};

cbuffer gPerPassConstants
{
    uint2 constants;
};

StructuredBuffer<BokehSplat> gBokehBuffer;

struct VSIn
{
    float2 pos : POSITION;
};

float4 packedHalf4ToFloat4(uint2 packedInput)
{
    float4 unpackedOutput;
    unpackedOutput.x = f16tof32(packedInput.x & 0x0000ffff).x;
    unpackedOutput.y = f16tof32(packedInput.x >> 16).x;
    unpackedOutput.z = f16tof32(packedInput.y & 0x0000ffff).x;
    unpackedOutput.w = f16tof32(packedInput.y >> 16).x;
    return unpackedOutput;
}

VertexToPixel splatVS(VSIn vsIn, uint instanceIndex: SV_InstanceID, uint vertexIndex: SV_VertexID)
{
    BokehSplat bokeh = gBokehBuffer[instanceIndex];
    VertexToPixel vsOut;

    uint2 screenPosI = uint2(bokeh.packedScreenPos & 0xFFFF, bokeh.packedScreenPos >> 16);
    float2 screenPos = (float2)screenPosI * .5f;

    // Just scale, 0 means no anisotropy, 1 means inf (it clamped to .95f from the host).
    const float anisoScale = 1.f / (1.f - abs(cameraAnisotropy));
    float2 anisotropyMult = float2(1.f / anisoScale, anisoScale);
    if (cameraAnisotropy < 0.f)
    {
        anisotropyMult = anisotropyMult.yx;
    }

    float radius = bokeh.getRadius();

    vsOut.position.xy = mad(vsIn.pos * anisotropyMult, abs(radius), screenPos);
    vsOut.position.xy = mad(vsOut.position.xy, invInputSize * float2(2, -2), float2(-1, 1));
    vsOut.position.zw = float2(1, 1);
    vsOut.color = packedHalf4ToFloat4(bokeh.color);
    vsOut.radius = abs(radius) * (1 << constants.x);
    vsOut.xy = vsIn.pos;

    return vsOut;
}

void splatPS(VertexToPixel input, out float4 output: SV_Target0)
{
    // Note: Original comments from OV, not all relevant to Falcor but leaving for future reference.
    // TODO (Max): Change blending to atomic ops, it's fp16 add.
    // Besides perf gain, this will also allow to forgo multiple draw calls and changing render targets,
    // and instead do all splatting in a single ExecuteIndirect.
    // Problem is, Vulkan doesn't support it.
    // NvInterlockedAddFp16x4 can probably be used for SPIRV target as it just writes to a UAV, but Vulkan itslef has no core mechanism to
    // change root parameters. There's NV extension for it though, but carb probably doesn't have it implemented.
    // TODO: Change this back to a clip() intrinsic when we add support for it in Slang-generated GLSL code.
    // clip(1 - length(input.xy));
    float distance = length(input.xy) * input.radius;
    float fadeStartRadius = max(0.f, input.radius - max(1.f, input.radius * 0.2f));
    float intensity = 1.f - smoothstep(fadeStartRadius, input.radius, distance); // Make the circle soft
    if (intensity == 0.f)
        discard;
    output = input.color;
}
