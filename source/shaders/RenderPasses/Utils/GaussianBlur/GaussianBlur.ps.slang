/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 # Copyright 2024 The Khronos Group, Inc.
 **************************************************************************/
#ifdef _USE_TEX2D_ARRAY
Texture2DArray gSrcTex;
#else
texture2D gSrcTex;
#endif

SamplerState gSampler;

struct BlurPSIn
{
    float2 texC : TEXCOORD;
    float4 pos : SV_POSITION;
#ifdef _USE_TEX2D_ARRAY
    uint arrayIndex : SV_RenderTargetArrayIndex;
#endif
};

Buffer<float> weights;

#ifdef _USE_TEX2D_ARRAY
float4 blur(float2 texC, uint arrayIndex)
#else
float4 blur(float2 texC)
#endif
{
#ifdef _HORIZONTAL_BLUR
    const float2 dir = float2(1, 0);
#elif defined _VERTICAL_BLUR
    const float2 dir = float2(0, 1);
#else
#error Please define either _HORIZONTAL_BLUR or _VERTICAL_BLUR
#endif

    const int2 offset = -(_KERNEL_WIDTH / 2) * dir;

    float4 c = float4(0, 0, 0, 0);

    [ForceUnroll]
    for (int i = 0; i < _KERNEL_WIDTH; i++)
    {
#ifdef _USE_TEX2D_ARRAY
        c += gSrcTex.SampleLevel(gSampler, float3(texC, arrayIndex), 0, offset + i * dir) * weights[i];
#else
        c += gSrcTex.SampleLevel(gSampler, texC, 0, offset + i * dir) * weights[i];
#endif
    }
    return c;
}

float4 main(BlurPSIn pIn) : SV_TARGET0
{
    float4 fragColor = float4(1.f, 1.f, 1.f, 1.f);

#ifdef _USE_TEX2D_ARRAY
    fragColor = blur(pIn.texC, pIn.arrayIndex);
#else
    fragColor = blur(pIn.texC);
#endif
    return fragColor;
}
