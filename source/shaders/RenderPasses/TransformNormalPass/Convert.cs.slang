/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/

/** Convert pass for Export Pass
 */

cbuffer PerFrameCB
{
    float4x4 gViewIT;
    float3 gCamWorldPos;
    uint2 gResolution;
}

Texture2D<float4> gNormal;
Texture2D<float4> gWorldPos;
RWTexture2D<float4> gOutput;

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (any(dispatchThreadId.xy >= gResolution))
        return;

    const uint2 pixelPos = dispatchThreadId.xy;

    float3 normal = gNormal[pixelPos].xyz;
    float dW = length(gWorldPos[pixelPos].xyz - gCamWorldPos);
    float depth = dW / (dW + 1.0f); // rcp(length(gWorldPos[pixelPos].xyz - gCamWorldPos));
    float3 viewNormal = normal;

    if (length(normal) > 0.01)
    {
        viewNormal = mul(gViewIT, float4(normal, 0.0f)).xyz;
        viewNormal = (normalize(viewNormal) + float3(1, 1, 1)) * 0.5f;
    }

    gOutput[pixelPos] = float4(viewNormal.xyz, depth);
}
